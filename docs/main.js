// Lightbox for figures, the emission off/on slider and copy buttons for the code blocks.

(function () {
  var lb = document.getElementById('lb');
  var lbImg = lb.querySelector('img');
  var opener = null;

  function open(img) {
    opener = img;
    lbImg.src = img.src;
    lbImg.alt = img.alt;
    lb.classList.add('on');
    document.body.style.overflow = 'hidden';
    lb.focus();
  }

  function close() {
    if (!lb.classList.contains('on')) return;
    lb.classList.remove('on');
    lbImg.src = '';
    document.body.style.overflow = '';
    if (opener) { opener.focus(); opener = null; }
  }

  // The figures are plain <img>, so make each one a real, focusable control.
  document.querySelectorAll('img.fig').forEach(function (img) {
    img.tabIndex = 0;
    img.setAttribute('role', 'button');
    img.setAttribute('aria-label', 'Enlarge figure' + (img.alt ? ': ' + img.alt : ''));
    img.addEventListener('click', function () { open(img); });
    img.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(img); }
    });
  });

  lb.tabIndex = -1;
  lb.setAttribute('role', 'dialog');
  lb.setAttribute('aria-modal', 'true');
  lb.addEventListener('click', close);
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') close();
  });

  // navigator.clipboard is undefined outside a secure context, and writeText can
  // reject; either way the button must say something rather than fail silently.
  function copy(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.style.cssText = 'position:fixed;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (err) { ok = false; }
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error('copy unavailable'));
    });
  }

  // Emission off/on comparisons: the divider follows a press-and-drag (the container sets
  // touch-action: pan-y pinch-zoom, so a vertical swipe still scrolls the page) and the arrow keys.
  // A round handle on the divider and an "Off" / "On" label on each side of it travel with it,
  // so the line reads as something to drag.
  // The slider role sits on the divider, not on the container: a slider's children are
  // presentational, which would hide the images' alt text from screen readers.
  // A container with .cmp__pane children (the results grid) shares one position across all its
  // panes: each pane draws its own line, the first also the handle, and a drag in any pane moves
  // them all, measured against the pane it started in.
  document.querySelectorAll('.cmp').forEach(function (el) {
    var pct = 50;
    var panes = Array.prototype.slice.call(el.querySelectorAll('.cmp__pane'));
    var grid = panes.length > 0;
    if (!grid) panes = [el];
    var ref = panes[0];
    var bar = document.createElement('span');
    bar.className = 'cmp__bar';
    bar.tabIndex = 0;
    bar.setAttribute('role', 'slider');
    bar.setAttribute('aria-label', el.dataset.label || 'Emission off on the left, on on the right');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '100');
    bar.innerHTML = '<span class="cmp__lab cmp__lab--off" aria-hidden="true">Off</span>' +
      '<span class="cmp__knob" aria-hidden="true">&lsaquo;&thinsp;&rsaquo;</span>' +
      '<span class="cmp__lab cmp__lab--on" aria-hidden="true">On</span>';
    panes[0].appendChild(bar);
    panes.slice(1).forEach(function (p) {
      var line = document.createElement('span');
      line.className = 'cmp__bar';
      line.setAttribute('aria-hidden', 'true');
      p.appendChild(line);
    });
    function set(p) {
      pct = Math.max(0, Math.min(100, p));
      el.style.setProperty('--x', pct + '%');
      bar.setAttribute('aria-valuenow', Math.round(pct));
      bar.setAttribute('aria-valuetext', Math.round(pct) + '% emission off');
    }
    function fromEvent(e) {
      var r = ref.getBoundingClientRect();
      set((e.clientX - r.left) / r.width * 100);
    }
    set(pct);
    // Track the one pointer that started on a render: a touch also captures implicitly, so a press
    // in a gap between cells must neither start a drag nor end one already going.
    var active = null;
    el.addEventListener('pointerdown', function (e) {
      if (grid) {
        var pane = e.target.closest('.cmp__pane');
        if (!pane) return;  // a heading, an input image or a gap, not a render
        ref = pane;
      }
      active = e.pointerId;
      el.setPointerCapture(e.pointerId);
      fromEvent(e);
    });
    el.addEventListener('pointermove', function (e) {
      if (e.pointerId === active) fromEvent(e);
    });
    ['pointerup', 'pointercancel'].forEach(function (t) {
      el.addEventListener(t, function (e) { if (e.pointerId === active) active = null; });
    });
    bar.addEventListener('keydown', function (e) {
      var step = e.shiftKey ? 10 : 2;
      if (e.key === 'ArrowLeft') set(pct - step);
      else if (e.key === 'ArrowRight') set(pct + step);
      else if (e.key === 'Home') set(0);
      else if (e.key === 'End') set(100);
      else return;
      e.preventDefault();
    });
  });

  document.querySelectorAll('.copy').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var was = btn.dataset.label || btn.textContent;
      btn.dataset.label = was;
      copy(document.querySelector(btn.dataset.copy).innerText).then(
        function () { btn.textContent = 'Copied'; },
        function () { btn.textContent = 'Press Ctrl+C'; }
      ).then(function () {
        setTimeout(function () { btn.textContent = was; }, 1600);
      });
    });
  });
})();
