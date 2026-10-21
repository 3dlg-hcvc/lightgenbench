# Emissive-74k Annotation — Guide

## What you're doing

For each 3D object, decide whether it's a **good emissive object** — something that emits
light from the part that *should* glow — then add a couple of labels. It doesn't have to be a
real-world light: it can be **realistic** (lamp, car headlights/taillights, screen, glowing
sign) **or artistic / stylized** (a glowing magic sword, a fantasy crystal, neon, a cartoon
glow). Both are good — you'll mark which under **style**.

**Before you decide — and whenever you're unsure — hover ⤢ more** and check the **emission
map (UV)** and **multi-view clips**; the card renders alone can mislead.

## Setup

1. **Set your name**
   ![The set button, next to "annotator:" in the top bar](static/set_name0.png)
   ![Typing your name into the box that pops up](static/setname_0.png)
2. **Claim a batch** Pick or enter the number of objects you want to label in this session.
   Then pick the categories you want to work in. If you want a mix of everything, then
   don't click on anything.
   ![Claim a batch: pick a size, then tick the categories you want](static/claimbatch1.png)
3. **Navigate.** Move between objects with the **← / →** arrow keys (or the buttons at the bottom of the screen).
   ![Paging between objects with the arrow buttons or the arrow keys](static/navigate0.png)

## What counts as emission

This is where almost all the hard calls live, so read it once carefully.

**A bright surface is not emission.** Pale, white, glossy, metallic, neon-colored and
vividly painted **materials** are not light-emitters — they just look bright under the scene
light. Count something as emission only if it still glows when nothing else is lighting it.

**The dark-box pair is the first test.** The two dark-box renders are the same scene with the
same faint fill light, differing *only* in whether the object's own emission is switched on. So
whatever appears between **off** and **on** *is* the emission — no matter how bright the object
looks in the normal render. But it is not the last word: the fill light can swamp a weak
emitter. **If the pair looks flat, check the emit-only views before deciding nothing glows** —
they contain nothing but the emission, so a dim glow shows up there when the dark box misses it.

**Stylized counts.** A magic sword, a fantasy crystal, neon, a cartoon glow — all genuine
emission, all **No issue**, style **artistic**. What matters is that it really glows, not
that it is a real-world lamp.

## Making a decision

**✓ No issue** — good emissive object: light comes from the parts that *should* glow, **and**
it actually shows up in the renders. Then set:

- **style** — **real** or **artistic**.
- **emission complete** (default on) — leave **checked** if every part that should glow does;
  **uncheck** if it's the right *kind* of object but some expected parts are missing (e.g. a
  car with only the dashboard lit). Missing parts are still **No issue** — just uncheck the
  box, don't use Has issue.

![A "No issue" decision — the green button, the style row, and the emission-complete box](static/no_issue.png)

**✗ Has issue** — Pick **one or more** reasons:

- **not emissive object** — not a light-emitter at all. Use this when the object is only
  bright *material* and the dark-box pair shows no real glow.
- **wrong emissive maps** — the emissive part does not make sense. Two common shapes:
  - **the whole object glows** — emission covers essentially the entire surface instead of
    only the bulb / screen / headlight / rune / neon tube. This is a bake bug, **not** a
    light fixture, and it is common — don't rationalize a fully-lit object as "a coherent glow".
  - **the glow is in the wrong place** — e.g. the lampshade lights up instead of the bulb,
    or emission sits on an unrelated patch of the surface.
- **multiple object** — a **scene, set or pack**: several distinct items arranged or bundled
  together. A single character **together with props it holds, wears or stands on is one
  object — that is not this flag.**
- **zero emission map** — the emission map in UV space is **completely black**. Faint, small
  or few emitting regions are *not* zero. This should be rare; fully-black shapes are
  filtered out before you see them.
- **anything else** — leave a note in the box. The most common case with no tag of its own:
  **the emission map has content but the renders don't show it** — the dark-box pair barely
  changes, or the object plainly renders wrong / too dark. Note it as
  *"renders don't show the emission"*.

## Quick decision tree

- Nothing changes between dark-box **off** and **on** — **and** the emit-only views are dark
  too → **Has issue**
  → *not emissive object* if it was never a light-emitter, otherwise
  *anything else* + note *"renders don't show the emission"*
- Dark-box barely changes **but** the emit-only views show a faint glow → that **is** emission.
  The dark box has a fill light that can swamp a weak emitter; the emit-only view has nothing
  else in it. Judge it on *where* the glow is, not on the dark box alone.
- Essentially the whole object glows → **Has issue → wrong emissive maps**
- The glow is real but in the wrong place → **Has issue → wrong emissive maps**
- Several separate items bundled together → **Has issue → multiple object**
  (a character holding a prop is **one** object — not this)
- Glows in the right place, and it shows in the renders → **No issue**, complete **checked**
- Same, but a few expected parts stay dark → **No issue**, complete **unchecked**
- Can't tell → **Has issue** + a note saying why

## The views

Four renders per object:

- **rendered object** — the object under normal lighting.
- **emissive-brightened** — the object rendered with a stronger emissive strength for its emissive part.
- **dark-box (emission off)** / **dark-box (emission on)** — the object in the same dark room,
  lit only by a very dim fill light. The two differ *only* in whether the object's own emission
  is switched on — so whatever changes between them *is* the emission.

**⤢ more** adds 4 additional renderings:

- **emit-only rendering** — only the emissive part of the object is rendered.
- **emission map (UV)** — the emissive map in UV space.
- **multi-view orbit** — the multi-view rendering of the object.
- **multi-view emit-only** — the multi-view rendering of the emissive part.

**⬡ Sketchfab** (same row) opens the original interactive model on Sketchfab.

![Hovering ⤢ more opens the extra views; ⬡ Sketchfab opens the source asset. Below, a "Has issue" decision with its reason picked](static/has_issue_example.png)

## Your batch

The top bar shows `batch: 12 / 50 done`. When it reaches the end the button changes to
**✓ batch done — claim another** — claim a new one whenever you're ready.

## Notes

- Everything **auto-saves** — no save button.
