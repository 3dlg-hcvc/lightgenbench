// Lightbox for figures and copy buttons for the code blocks.

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
