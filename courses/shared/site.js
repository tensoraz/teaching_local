// Shared behaviour for course pages: theme toggle, reading progress, TOC highlight, copy buttons.
(function () {
  var root = document.documentElement;
  try {
    var saved = localStorage.getItem('course-theme');
    if (saved) root.setAttribute('data-theme', saved);
  } catch (e) {}

  window.toggleTheme = function () {
    var dark = root.getAttribute('data-theme') === 'dark' ||
      (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
    var next = dark ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('course-theme', next); } catch (e) {}
  };

  document.addEventListener('DOMContentLoaded', function () {
    var bar = document.querySelector('.progress');
    var links = Array.prototype.slice.call(document.querySelectorAll('.toc a[href^="#"]'));
    var targets = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); });

    function onScroll() {
      var h = document.documentElement;
      if (bar) bar.style.width = (100 * h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight)) + '%';
      var current = 0;
      targets.forEach(function (t, i) { if (t && t.getBoundingClientRect().top < 120) current = i; });
      links.forEach(function (a, i) { a.classList.toggle('on', i === current); });
    }
    addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    var esc = function (s) { return s.replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); };
    var tok = /(#[^\n]*)|("""[\s\S]*?"""|f?"(?:[^"\\\n]|\\.)*"|f?'(?:[^'\\\n]|\\.)*')|\b(import|from|as|def|return|for|in|if|else|elif|while|True|False|None|and|or|not|with|lambda|class|is)\b|\b(\d+\.?\d*(?:e\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g;
    document.querySelectorAll('pre.code[data-lang="py"]').forEach(function (pre) {
      var src = pre.textContent, out = '', last = 0, m;
      tok.lastIndex = 0;
      while ((m = tok.exec(src))) {
        out += esc(src.slice(last, m.index));
        var cls = m[1] ? 'c' : m[2] ? 's' : m[3] ? 'k' : m[4] ? 'n' : 'f';
        out += '<span class="' + cls + '">' + esc(m[0]) + '</span>';
        last = tok.lastIndex;
      }
      pre.innerHTML = out + esc(src.slice(last));
    });

    document.querySelectorAll('pre.code').forEach(function (pre) {
      var b = document.createElement('button');
      b.className = 'copy';
      b.textContent = 'Copy';
      b.onclick = function () {
        var text = Array.prototype.filter.call(pre.childNodes, function (n) { return n !== b; })
          .map(function (n) { return n.textContent; }).join('');
        var done = function () { b.textContent = 'Copied ✓'; setTimeout(function () { b.textContent = 'Copy'; }, 1500); };
        if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(text).then(done);
        else {
          var ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta);
          ta.select(); try { document.execCommand('copy'); } catch (e) {} ta.remove(); done();
        }
      };
      pre.appendChild(b);
    });

    document.querySelectorAll('.flip').forEach(function (f) {
      f.addEventListener('click', function () { f.classList.toggle('on'); });
    });
  });
})();
