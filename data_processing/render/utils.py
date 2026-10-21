"""What the review-image renderers (`dark_box.py`, `orbit.py`, `single_view.py`) share: the
paths and settings, the per-shape guard, the bpy render helper, and the coverage scan. Where
each image is lives in `utils/review_images.py`.

bpy, PIL and pandas are imported inside the functions that use them, so importing this module
loads nothing heavy. The coverage scan is its one command:

    python -m data_processing.render.utils coverage [--list <uuids>] ...
"""

from __future__ import annotations

import argparse
import errno
import os
import random
import signal
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from utils import review_images
from utils.paths import (
    ALBEDO_COPY_PASS_LIST,
    TEXVERSE_1K_ROOTS,
    TEXVERSE_METADATA_PARQUET,
)
from utils.review_images import BOX_ROOT  # noqa: F401  (re-exported)


# ====================================================================== paths and settings
# The GLB and metadata locations come from `utils/paths.py`, as the bake's do. The image roots
# and the lookups are in `utils/review_images.py`.

#: The bake's input: the heuristic stage's pass list, one uuid per line.
INPUT_LIST = ALBEDO_COPY_PASS_LIST
#: Metadata parquet with `glb_1k_path` and `ditem_dir` for every downloaded GLB.
DEFAULT_METADATA = TEXVERSE_METADATA_PARQUET
DEFAULT_GLB_ROOTS = TEXVERSE_1K_ROOTS

HERE = Path(__file__).resolve().parent

#: Local, gitignored: the review lanes' own outputs.
OUT_DIR = HERE / "out"
#: Run records, one directory per run.
RUNS_DIR = OUT_DIR / "runs"
#: The coverage parquet, the work lists, the shard cuts.
REVIEW_OUT_DIR = OUT_DIR / "review"

# --- what a render is ---------------------------------------------------------------
# The image kinds a lookup takes (`STILL_KINDS`, `ORBIT_PRESETS`) are in `utils/review_images.py`.
#: Frame order: Front, Right, Back, Left, Top, Bottom. The packer writes frames in this
#: order and the annotator's orbit plays it; changing it changes every webp.
VIEWS = ("-y_to_+y", "+x_to_-x", "+y_to_-y", "-x_to_+x", "+z_to_-z", "-z_to_+z")
#: The label burnt into each frame, one per view in the same order. Every shipped webp
#: carries them and the frames are read uncropped, so a new shape's orbit must carry them
#: too. Next to `VIEWS` so the two orders cannot drift apart.
VIEW_LABELS = ("Front", "Right", "Back", "Left", "Top", "Bottom")
#: How that label is drawn, reproducing the shipped frames: white text on a translucent
#: black box centered at the top.

#: The label font. When this path is absent Pillow also searches the system font
#: directories for the file name; a machine without the font at all raises.
LABEL_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LABEL_FONT_SIZE = 32
#: Distance from the top of the frame to the top of the text, and the box's margin
#: around it.
LABEL_TOP = 20
LABEL_PAD = 10
LABEL_BOX_FILL = (0, 0, 0, 180)
LABEL_TEXT_FILL = (255, 255, 255, 255)
#: Camera position and up vector per view, target the origin. 2*sqrt(3) is the default
#: (2,2,2) distance.
_D = 3.4641016151377544
VIEW_CAMERA = {
    "-y_to_+y": ((0, -_D, 0), (0, 0, 1)),
    "+x_to_-x": ((_D, 0, 0), (0, 0, 1)),
    "+y_to_-y": ((0, _D, 0), (0, 0, 1)),
    "-x_to_+x": ((-_D, 0, 0), (0, 0, 1)),
    "+z_to_-z": ((0, 0, _D), (0, 1, 0)),
    "-z_to_+z": ((0, 0, -_D), (0, -1, 0)),
}
#: Blend asset name per still kind and per orbit preset. The lit still and the lit orbit
#: are the same blend; only the camera differs (the still leaves the blend's own camera
#: alone, the orbit overrides it).
PRESET_BLEND = {
    "glb": "preset_glb.blend",
    "emit": "preset_glb_emit_only.blend",
    "brighter": "preset_glb_emission_brighter.blend",
    "standard": "preset_glb.blend",
    "emit_only": "preset_glb_emit_only.blend",
}
#: The preset blends, tracked beside this module. Every bpy render in the repo starts from
#: one of them (`render_emissive.py` loads the lit one as `PRESET_GLB`).
BLEND_ASSETS_DIR = HERE / "assets"
REVIEW_RESOLUTION = (512, 512)
#: 8 samples + denoise reproduces the shipped emit-only still byte for byte.
REVIEW_SAMPLES = 8
REVIEW_DENOISE = True


# ====================================================================== the per-shape guard
# For the still and orbit workers: a wall-clock cap, a crash marker and a failures file, so
# one bad GLB costs one attempt and is then named and skipped (`dark_box.py` follows the same
# pattern). A hang inside C code, where a Python signal cannot fire, needs the process killed
# from outside; the marker below then skips the shape.
#
#     guard = ShapeGuard(guard_dir, rank, timeout_s)
#     skip = guard.adopt()                    # crashed last time or failed before: skipped
#     for uuid, ... in rows:
#         if uuid in skip:
#             continue
#         with guard.attempt(uuid) as att:
#             ... render every view ...
#         if att.failed:
#             ...                             # named in failures_rank<r>.txt, never retried
#
# A marker `<uuid>.inprogress.<rank>` exists exactly while a shape renders. If the process
# dies (OOM, a SIGKILL, a bpy segfault) the marker survives and the next process
# on this rank adopts it as `crashed_process`. Views that landed before the death stay on
# disk - resume is by file existence - so they are not redone.
#
# `ShapeTimeout` derives from BaseException on purpose: `render_view` / `render_one` catch
# `Exception` so one bad view does not take the shard down, and a timeout caught there
# would silently disarm the cap for the shape's remaining views.

class ShapeTimeout(BaseException):
    """The per-shape wall-clock cap fired. Not an Exception: see the section head."""


# A write the store refuses for space or quota stops the shard: counting
# it as the shape's failure would name every shape left in the shard a failure and skip them
# all on the rerun. Unrecorded, the shape is retried once space is freed. Blender reports a
# refused write as text, not as an errno, hence the message check.
_DISK_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT})
_DISK_FULL_TEXT = ("No space left on device", "Disk quota exceeded")


def is_disk_full(exc: BaseException) -> bool:
    """True for a write refused because the filesystem is full or the quota is used up."""
    if isinstance(exc, OSError) and exc.errno in _DISK_FULL_ERRNOS:
        return True
    return any(t in str(exc) for t in _DISK_FULL_TEXT)


#: Below this much free space a file Blender left missing or empty counts as refused.
_LOW_SPACE_BYTES = 64 << 20


def check_written(path) -> None:
    """Raise a disk-full OSError if Blender left `path` missing or empty on a nearly full
    filesystem: Blender reports a refused write as a console message, not an exception."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return
    st = os.statvfs(path.parent)
    if st.f_bavail * st.f_frsize < _LOW_SPACE_BYTES:
        raise OSError(errno.ENOSPC, f"{path} was not written and its filesystem is full")


def append_line(path, line: str) -> None:
    """Append one line; if the store refuses it, cut the file back so no half line is left
    (a resume reader would take a half line naming a uuid as that shape's failure).

    Unbuffered on purpose: a buffered file flushes before it truncates, and on a store that
    is still full that flush fails again and the cut never happens."""
    data = line.encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        start = os.fstat(fd).st_size
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
        except OSError as e:
            if is_disk_full(e):
                os.ftruncate(fd, start)
            raise
    finally:
        os.close(fd)


class Attempt:
    __slots__ = ("uuid", "failed", "error", "seconds")

    def __init__(self, uuid: str):
        self.uuid, self.failed, self.error, self.seconds = uuid, False, None, 0.0


class ShapeGuard:
    def __init__(self, guard_dir, rank: int, timeout_s: int):
        self.dir = Path(guard_dir)
        self.rank = int(rank)
        self.timeout_s = int(timeout_s)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.failures = self.dir / f"failures_rank{self.rank}.txt"
        self._suffix = f".inprogress.{self.rank}"

    def marker(self, uuid: str) -> Path:
        return self.dir / f"{uuid}{self._suffix}"

    def record(self, uuid: str, reason: str) -> None:
        append_line(self.failures, f"{uuid}\t{reason}\n")

    def adopt(self) -> set:
        """Uuids this rank must skip: crashed mid-render last time, or failed before.

        A leftover marker is consumed here, so a crash is recorded once and the shape is
        then skipped by the failures file like any other failure.
        """
        crashed = []
        for p in sorted(self.dir.glob(f"*{self._suffix}")):
            crashed.append(p.name[: -len(self._suffix)])
            p.unlink()
        for u in crashed:
            self.record(u, "crashed_process")
        skip = set(crashed)
        if self.failures.exists():
            skip |= {ln.split("\t", 1)[0]
                     for ln in self.failures.read_text().splitlines() if ln.strip()}
        return skip

    def _alarm(self, signum, frame):
        raise ShapeTimeout(f"{self.timeout_s}s")

    @contextmanager
    def attempt(self, uuid: str):
        att = Attempt(uuid)
        m = self.marker(uuid)
        m.touch()
        prev = signal.signal(signal.SIGALRM, self._alarm)
        signal.alarm(self.timeout_s)
        t0 = time.time()
        try:
            yield att
        except (ShapeTimeout, Exception) as e:   # noqa: BLE001 - one bad GLB, not the shard
            if is_disk_full(e):
                # The store is full: stop the shard, record nothing (the marker goes in
                # `finally`), so the rerun after freeing space renders this shape again.
                print(f"[guard] {uuid}: the store refused a write ({e}); stopping the shard",
                      flush=True)
                raise
            att.failed, att.error = True, f"{type(e).__name__}: {e}"
            self.record(uuid, att.error)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, prev)
            att.seconds = time.time() - t0
            m.unlink(missing_ok=True)


# ====================================================================== the bpy render helper
# The repo's one bpy render helper: the review lanes and the two dark-room renderers use it.
# A render needs only bpy, numpy and PIL; bpy and PIL are imported inside the functions, so
# a module that imports this one for its paths never loads them.
#
# **Byte-for-byte parity is the contract.** The `emit` still matches the shipped emit-only
# render by sha256, and the orbit and `brighter` renders match theirs within one 8-bit level.
# Two details carry that parity and are a matched pair - change one and the other stops
# meaning what it means:
#
# 1. `read_img` reads the 16-bit render Blender writes through PIL, which keeps the HIGH
#    BYTE of each sample, and divides by 255 in float32. Values therefore land exactly on
#    k/255.
# 2. `save_img` quantizes with `* 255` then `.astype(np.uint8)` - truncation, not
#    rounding - which is what the producer of the shipped renders did.
#
# On k/255 values the two quantizers agree, so today only (1) is load-bearing; a read that
# preserved the full 16 bits would make (2) load-bearing instead, on most pixels of every
# frame.

# ------------------------------------------------------------------ image IO
def read_img(path):
    """Read a render as float32 in [0, 1], exactly as `matplotlib.image.imread` reads an
    RGBA PNG: PIL reduces a 16-bit file to its high byte, and the division by 255 puts every
    value on k/255 (see the section head).

    Unlike `mpimg.imread`, a path that is not a PNG also comes back as float 0-1, not uint8.
    """
    from PIL import Image

    path = os.path.expanduser(path)
    with Image.open(path) as img:
        img.load()
        if img.mode not in ("RGB", "RGBA"):
            # `render_scene` always writes RGBA; anything else means the caller handed us
            # a file this path did not produce, and matplotlib would have scaled it by a
            # different divisor.
            raise ValueError(f"Unsupported PIL mode for a render: {img.mode}")
        return np.divide(np.asarray(img), 2 ** 8 - 1, dtype=np.float32)


def save_img(target, img):
    """Fast image saving using PIL with RGBA support.
    Assume value range is [0, 1] for float input types, and [0, 255] for int input types.

    Args:
        target: Path to save the image
        img: numpy array (H, W, C) or (H, W) where C can be 1, 3, or 4 channels
    """
    from PIL import Image

    # Handle float dtype
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0., 1.) * 255
        img = img.astype(np.uint8)          # TRUNCATION, not rounding: see the section head
    elif img.dtype != np.uint8:
        # Convert other integer types to uint8 if needed
        img = img.astype(np.uint8)

    # Determine the mode based on shape. PIL infers the same mode from dtype and shape,
    # so it is not passed to `fromarray` (deprecated there, removed in Pillow 13); the
    # check stays because it is what rejects a bad channel count with a clear message.
    if len(img.shape) == 2:
        pass                                                    # grayscale, 'L'
    elif len(img.shape) == 3:
        if img.shape[2] == 1:
            img = img.squeeze(axis=2)                           # 'L'
        elif img.shape[2] not in (3, 4):                        # 'RGB' / 'RGBA'
            raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    Image.fromarray(img).save(target)
    return img


# ------------------------------------------------------------------ scene data
def load_blend(path, verbose=False):
    """Open a .blend file, replacing the current scene. The caller names the asset
    (`BLEND_ASSETS_DIR`)."""
    import bpy

    if path[0] != '/' and path[0] != '~':
        print("Warning: path is not absolute, trying to convert to absolute path.")
        path = os.path.join(os.getcwd(), path)
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    if verbose:
        print("Clearing everything and loading blend file: {}".format(path))
    bpy.ops.wm.open_mainfile(filepath=path)


def load_glb(glb_path, collection='workbench',
             record_mesh_index=True,
             maximum_number_of_meshes=-1,
             import_shading='NORMALS', merge_vertices=False, apply_armature=True):
    ''' Load a glb file and return the object.
        Note that a glb loaded may contain multiple meshes.
    '''
    import bpy

    bpy.ops.object.select_all(action='DESELECT')
    extra_kwargs = dict()
    if import_shading is not None:
        extra_kwargs = dict(import_shading=import_shading)
    if apply_armature == True:
        # It is important to disable bone shape and guess original bind pose as it may cause issues with the mesh.
        extra_kwargs.update(dict(disable_bone_shape=True, guess_original_bind_pose=False))
    bpy.ops.import_scene.gltf(filepath=glb_path, merge_vertices=merge_vertices, **extra_kwargs)
    # join all selected objects
    num_meshes = 0
    if maximum_number_of_meshes != -1 and len(bpy.context.selected_objects) > maximum_number_of_meshes:
        raise ValueError(f"Exceeded maximum number of meshes. {len(bpy.context.selected_objects)} > {maximum_number_of_meshes}")

    for obj in bpy.context.selected_objects:
        # Check if the object is not a mesh
        if obj.type == 'MESH':  # set a mesh object as active (in order to join)
            bpy.context.view_layer.objects.active = obj
            if record_mesh_index == False:
                break
            # create a new face attribute of the mesh storing the obj_index
            glb_mesh_index = np.zeros(len(obj.data.polygons), dtype=np.float32) + num_meshes
            bpy.ops.geometry.attribute_add(name="glb_mesh_index_original", domain='FACE', data_type='FLOAT')
            obj.data.attributes['glb_mesh_index_original'].data.foreach_set('value', glb_mesh_index + 1)
            num_meshes = num_meshes + 1
    bpy.ops.object.join()
    if record_mesh_index == False:
        glb_mesh_index = np.zeros(len(obj.data.polygons), dtype=np.float32) + num_meshes
        bpy.ops.geometry.attribute_add(name="glb_mesh_index_original", domain='FACE', data_type='FLOAT')
        obj.data.attributes['glb_mesh_index_original'].data.foreach_set('value', glb_mesh_index + 1)
        num_meshes = num_meshes + 1

    # apply armature if it exists
    if apply_armature:
        obj = bpy.context.view_layer.objects.active
        # if it is an armature modifier, apply it
        for modifier in obj.modifiers:
            if modifier.type == 'ARMATURE':
                # Blender refuses modifier_apply on a mesh that carries shape keys, and
                # the join above gives the result shape keys if ANY source mesh had them.
                # The shape keys here are glTF morph targets sitting at weight 0, so the
                # visible geometry is the basis either way: drop them and apply, rather
                # than lose the shape. A mesh without shape keys is untouched.
                if obj.data.shape_keys is not None:
                    obj.shape_key_clear()
                bpy.ops.object.modifier_apply(modifier=modifier.name)

    obj = bpy.context.view_layer.objects.active
    normalize_mesh(obj)
    for obj in bpy.context.selected_objects:
        # remove non-mesh objects
        if obj.type != 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)
    # move all selected objects to the workbench collection
    move_selected_to_collection(collection)
    set_active_exclusive(obj)
    return obj


def move_selected_to_collection(collection_name):
    import bpy

    # Get the collection
    collection = bpy.data.collections.get(collection_name)
    if collection:
        # Loop through selected objects
        for obj in bpy.context.selected_objects:
            # Unlink the object from other collections
            for col in list(obj.users_collection):
                if col.name != collection_name:
                    col.objects.unlink(obj)
            # Link the object to the collection only if not already there
            if obj.name not in collection.objects:
                collection.objects.link(obj)
        # Update the scene
        bpy.context.view_layer.update()
    else:
        print(f"Collection '{collection_name}' not found. Create it now.")
        bpy.data.collections.new(collection_name)
        move_selected_to_collection(collection_name)


def set_active_exclusive(objs):
    import bpy

    if type(objs) is not list:
        objs = [objs]
    # if there is an active object
    if bpy.context.active_object is not None:
        bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objs:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj


def is_structrna_valid(obj):
    try:
        _ = obj.name
        return True
    except ReferenceError:
        return False


def purge_obj(obj):
    import bpy

    # if obj is already removed, then return
    if is_structrna_valid(obj) == False:
        orphan_purge()
        return

    bpy.data.objects.remove(obj, do_unlink=True)
    orphan_purge()


def orphan_purge():
    import bpy

    bpy.ops.outliner.orphans_purge(do_recursive=True)


def cleanup_unused_data():
    import bpy

    # Function to remove unused data blocks of a specific type
    def remove_unused_data_block(data_block_collection):
        for item in [block for block in data_block_collection if block.users == 0]:
            data_block_collection.remove(item)

    # List of data types to clean up
    data_types_to_cleanup = [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.curves,
        bpy.data.metaballs,
        bpy.data.fonts,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.libraries,
        bpy.data.actions,
        bpy.data.particles,
        # Add more data types here as needed
    ]

    # Remove unused data blocks of each type
    for data_type in data_types_to_cleanup:
        remove_unused_data_block(data_type)

    print("Recursive unused data blocks cleanup complete.")


def clear_collection(collection_name):
    import bpy

    # Get the collection
    collection = bpy.data.collections.get(collection_name)
    if collection:
        # Duplicate the list of objects since we'll be modifying it during iteration
        objs = list(collection.objects)
        # Deselect all objects
        bpy.ops.object.select_all(action='DESELECT')
        # Loop through objects and delete them
        for obj in objs:
            # if obj is rootnode, skip
            if 'RootNode' in obj.name:
                continue
            # Select the object
            obj.select_set(True)
            # Remove the object from the collection
            collection.objects.unlink(obj)
            # Delete the object
            bpy.data.objects.remove(obj)
        # Update the scene
        bpy.context.view_layer.update()
    else:
        print(f"Collection '{collection_name}' not found. Create it now.")
        bpy.data.collections.new(collection_name)
    cleanup_unused_data()


# ------------------------------------------------------------------ spatial transform
def normalize_mesh(obj):
    import bpy

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    # clear the glb's default axes tree structure.
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')

    # move the object origin (the yellow dot) to world origin.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # move the geometry to object (world) origin, so that the origin is the center of the bounding box.
    bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')

    # scale the object to fit the bounding box into a unit cube.
    max_extent = max(obj.dimensions)
    scale_factor = 1.9999 / max_extent if max_extent != 0 else 1
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return obj


# ------------------------------------------------------------------ camera
def set_camera_pos(location):
    import bpy

    # Get the current scene
    scene = bpy.context.scene
    # Ensure there is an active camera
    if not scene.camera:
        print("No active camera found in the scene.")
        return
    # Get the active camera
    camera = scene.camera
    # Set the camera location
    camera.location = location


def set_camera_orientation(location, target=(0, 0, 0), up=(0, 0, 1)):
    """Set camera position, point it at target, with specified up vector.

    Constructs proper camera orientation matrix from direction and up vectors,
    avoiding gimbal lock issues when looking straight up/down.

    `mathutils` is Blender's own math module and computes in single precision; the six
    orbit cameras must match the shipped views byte for byte, so do not re-derive this in
    numpy.
    """
    import bpy
    import mathutils

    scene = bpy.context.scene
    if not scene.camera:
        print("No active camera found in the scene.")
        return

    camera = scene.camera
    camera.location = location

    # Build rotation matrix from direction and up vectors
    # This is similar to a "look-at" matrix construction
    forward = (mathutils.Vector(target) - mathutils.Vector(location)).normalized()
    up_vec = mathutils.Vector(up).normalized()

    # Right vector = forward × up
    right = forward.cross(up_vec)

    # Handle parallel vectors (gimbal lock case)
    if right.length < 0.001:
        # forward and up are parallel, choose arbitrary perpendicular vector
        # Use world X or Y axis depending on which is more perpendicular
        if abs(forward.z) < 0.9:
            right = mathutils.Vector((0, 0, 1)).cross(forward)
        else:
            right = mathutils.Vector((1, 0, 0)).cross(forward)

    right.normalize()

    # Recalculate up to ensure orthogonality: up = right × forward
    up_vec = right.cross(forward)
    up_vec.normalize()

    # Build rotation matrix (Blender camera points down -Z, Y is up)
    # So: right → X, up → Y, -forward → Z
    rot_matrix = mathutils.Matrix((
        right,
        up_vec,
        -forward
    )).transposed().to_4x4()

    # Apply rotation
    camera.matrix_world = mathutils.Matrix.Translation(location) @ rot_matrix


# ------------------------------------------------------------------ render
def random_temp_filename(suffix='.png', temp_dir='/tmp/lightgen_blender_render/'):
    """A random file name in `temp_dir`, which `render_scene` writes, reads back and deletes.

    `exist_ok=True` because several ranks starting together on one machine can race
    between the `os.path.exists` and the `os.makedirs`.
    """
    name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    fname = name + suffix

    temp_dir = os.path.expanduser(temp_dir)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
    path = os.path.join(temp_dir, fname)
    return path, name


def render_scene(obj=None, resolution=(256,)*2, samples=1024, resolution_percentage=100,
                 shadow_catcher=True, render_engine='CYCLES', camera_position=None,
                 camera_up=None, output_path=None, denoise=True,
                 view_transform='Khronos PBR Neutral', view_look=f'None',
                 ):  # can not use BLENDER_EEVEE in docker
    import bpy

    if camera_position is not None:
        if camera_up is not None:
            set_camera_orientation(camera_position, target=(0, 0, 0), up=camera_up)
        else:
            set_camera_pos(camera_position)

    # Set render settings
    bpy.context.scene.view_layers["ViewLayer"].use_freestyle = False
    bpy.context.scene.render.film_transparent = True
    # Get the 'Floor' object
    floor = bpy.data.objects.get('Floor')
    if obj is not None:
        # get the lowest z value of the object to set the floor
        min_z = obj.bound_box[0][2]
        # set the floor to the lowest z value of the object
        floor.location.z = min_z - .01
    output_path_is_temp = False
    if output_path is None:
        file_format = 'PNG'
        output_path, _ = random_temp_filename(suffix='.png')
        print("output_path", output_path)
        output_path_is_temp = True
    else:
        # Ensure the output path exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        file_format = output_path.split('.')[-1].upper()
    if render_engine == 'CYCLES':
        print("default samples", bpy.context.scene.cycles.samples)
        bpy.context.scene.cycles.samples = samples
        bpy.context.scene.cycles.use_denoising = denoise
        # set the floor as shadow catcher
        if floor:
            floor.cycles.is_shadow_catcher = True
        # make it visible in render
        floor.hide_render = not shadow_catcher
        bpy.context.scene.view_settings.view_transform = view_transform
        bpy.context.scene.view_settings.look = view_look
    else:
        floor.hide_render = True

    bpy.context.scene.render.engine = render_engine
    # Cycles sizes its thread pool from the machine's core count, not the job's cgroup, so
    # under Slurm several workers oversubscribe the allocation. Cap to the allocation there
    # and only there: outside Slurm nothing changes, which keeps byte parity exact.
    cap = os.environ.get("SLURM_CPUS_PER_TASK", "")
    if cap.isdigit() and int(cap) > 0:
        bpy.context.scene.render.threads_mode = 'FIXED'
        bpy.context.scene.render.threads = int(cap)
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]
    bpy.context.scene.render.resolution_percentage = resolution_percentage
    print("File format: {}".format(file_format))
    bpy.context.scene.render.image_settings.file_format = file_format
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    #: 16 bits per channel. PIL keeps the high byte of each sample when `read_img` reads
    #: it back, which is what puts the returned floats on k/255. See the section head.
    bpy.context.scene.render.image_settings.color_depth = '16'
    # Set output path
    bpy.context.scene.render.filepath = output_path

    # Render the scene
    # enable compositing
    bpy.context.scene.use_nodes = True
    bpy.context.scene.render.use_compositing = True
    try:
        bpy.ops.render.render(write_still=True, layer="ViewLayer")
        print(f"Render complete. Image saved to: {bpy.context.scene.render.filepath}")
        check_written(output_path)       # a full /tmp: stop the shard (is_disk_full)
        img = read_img(output_path)
    finally:
        if output_path_is_temp and os.path.exists(output_path):
            os.remove(output_path)
    return img


def render_glb(glb_path, preset_blend, **kwargs):
    load_blend(preset_blend)
    clear_collection('workbench')
    obj = load_glb(glb_path, import_shading=None)
    img = render_scene(obj=obj, **kwargs)
    purge_obj(obj)
    return img


# ====================================================================== the coverage scan
# pandas and `utils.metadata_io` (which imports pandas) are
# imported inside `scan` and `coverage_main`, so a render worker never loads them; the
# `pd.DataFrame` annotations are text only (`from __future__ import annotations`).

COVERAGE_DOC = """What is missing, per image kind, over a uuid list.

This is the stage's own gate: it decides what the campaign renders, and re-running it
after the campaign is how the campaign is verified (counting files, not exit codes).
It only stats paths through `utils/review_images.py`'s lookups, so it is cheap enough to run
over the whole set (eight stats a shape, nearly all of the time NFS wait).
"""

KINDS = ("glb", "emit", "brighter", "orbit_standard", "orbit_emit_only", "box", "atlas")


def _row(uuid: str, shard: str) -> dict:
    ri = review_images
    on, off = ri.box(uuid)
    out = {"uuid": uuid, "shard": shard}
    for kind in ri.STILL_KINDS:
        out[kind] = bool(shard) and ri.still(kind, shard, uuid) is not None
    out["orbit_standard"] = ri.webp("standard", uuid) is not None
    out["orbit_emit_only"] = ri.webp("emit_only", uuid) is not None
    # Half a pair is a crashed render, not a pass.
    out["box"] = on is not None and off is not None
    out["atlas"] = ri.atlas(uuid) is not None
    return out


def _stat_roots() -> dict:
    """The roots `scan` stats. The NEW roots are deliberately absent: before a run
    writes its first file they do not exist yet, and that is not a mount failure."""
    ri = review_images
    return {
        **{f"LEGACY_STILL_ROOTS[{k!r}]": v
           for k, v in ri.LEGACY_STILL_ROOTS.items()},
        "LEGACY_ORBIT_ROOT": ri.LEGACY_ORBIT_ROOT,
        "LEGACY_WEBP_ROOT": ri.LEGACY_WEBP_ROOT,
        "BOX_ROOT": ri.BOX_ROOT,
        "ATLAS_ROOT": ri.ATLAS_ROOT,
    }


def assert_roots_mounted() -> None:
    """Refuse to scan against a root that is not there.

    `review_images._first_existing` asks `Path.exists()`, which returns False on ENOENT,
    which is also what an unmounted network share looks like. A scan against an unmounted
    root reports every shape missing and nothing refuses. The render lanes are bounded (each
    resumes against both roots), but this scan is the verdict.
    """
    missing = [f"{name} -> {root}"
               for name, root in _stat_roots().items()
               if not Path(root).is_dir()]
    if missing:
        raise RuntimeError(
            "review-image root is not a mounted directory:\n  " + "\n  ".join(missing)
            + "\nA scan against an unmounted root silently reports every shape missing, "
              "because Path.exists() returns False on ENOENT. Mount the export (or "
              "repoint the root) before scanning.")


def scan(uuids, metadata_path=None, workers=64) -> pd.DataFrame:
    """One row per uuid, in the order given. A uuid with no metadata row keeps its row
    with an empty shard and every kind False — it is missing, not absent."""
    import pandas as pd

    from utils.metadata_io import shard_of

    assert_roots_mounted()
    meta = pd.read_parquet(Path(metadata_path or DEFAULT_METADATA),
                           columns=["ditem_dir"])
    ditem = meta["ditem_dir"].to_dict()
    pairs = [(u, shard_of(ditem[u]) if isinstance(ditem.get(u), str) else "")
             for u in uuids]
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda p: _row(*p), pairs))
    return pd.DataFrame(rows, columns=["uuid", "shard", *KINDS])


def missing_lists(df: pd.DataFrame) -> dict:
    return {kind: df.loc[~df[kind].astype(bool), "uuid"].tolist() for kind in KINDS}


def summary(df: pd.DataFrame) -> str:
    lines = [f"shapes: {len(df)}"]
    for kind in KINDS:
        have = int(df[kind].astype(bool).sum())
        lines.append(f"  {kind:16s} have {have:6d}  missing {len(df) - have:6d}")
    complete = int(df[list(KINDS)].astype(bool).all(axis=1).sum())
    lines.append(f"complete on all kinds: {complete}  incomplete: {len(df) - complete}")
    return "\n".join(lines)


def coverage_main(argv=None) -> None:
    from utils.metadata_io import read_uuids, write_uuids

    ap = argparse.ArgumentParser(
        prog="python -m data_processing.render.utils coverage",
        description=COVERAGE_DOC)
    ap.add_argument("--list", default=str(INPUT_LIST))
    ap.add_argument("--out", default=str(REVIEW_OUT_DIR / "coverage.parquet"))
    ap.add_argument("--work_dir", default=str(REVIEW_OUT_DIR / "missing"),
                    help="one <kind>.txt of missing uuids is written here")
    ap.add_argument("--metadata", default=str(DEFAULT_METADATA))
    ap.add_argument("--workers", type=int, default=64)
    a = ap.parse_args(argv)

    uuids = read_uuids(a.list)
    df = scan(uuids, metadata_path=a.metadata, workers=a.workers)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    wd = Path(a.work_dir)
    wd.mkdir(parents=True, exist_ok=True)
    for kind, missing in missing_lists(df).items():
        write_uuids(wd / f"{kind}.txt", missing)
    print(summary(df))
    print(f"\ncoverage -> {a.out}\nwork lists -> {wd}")


def main(argv=None) -> None:
    """`coverage` is the one subcommand."""
    argv =sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["coverage"]:
        return coverage_main(argv[1:])
    ap = argparse.ArgumentParser(prog="python -m data_processing.render.utils",
                                 description="The review images' shared code. Its one command:")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("coverage", add_help=False,
                   help="what is missing, per image kind; `coverage -h` lists its flags")
    ap.parse_args(argv)


if __name__ == "__main__":
    main()
