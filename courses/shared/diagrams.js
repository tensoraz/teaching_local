// Small SVG helpers so diagrams stay crisp, themeable (CSS variables) and offline.
(function () {
  var NS = 'http://www.w3.org/2000/svg';

  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function ensureArrow(svg, id, color) {
    var defs = svg.querySelector('defs') || el('defs', {}, svg);
    if (svg.querySelector('#' + id)) return;
    var m = el('marker', { id: id, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 z', style: 'fill:' + color }, m);
  }

  function text(parent, x, y, str, style) {
    var t = el('text', { x: x, y: y, 'text-anchor': 'middle', style: style || '' }, parent);
    t.textContent = str;
    return t;
  }

  /**
   * Nodes placed on an ellipse, joined by curved arrows that follow the ellipse.
   * opts: {cx, cy, rx, ry, w, h, start (deg), gap (deg), items: [{t, s, fill, ink}], center: [line, line]}
   */
  window.drawCycle = function (svg, opts) {
    var n = opts.items.length, start = opts.start == null ? -90 : opts.start;
    var step = 360 / n, gap = opts.gap || step * 0.3;
    var arrowId = svg.id + '-arw';
    ensureArrow(svg, arrowId, 'var(--ink-3)');
    var pt = function (deg) {
      var r = deg * Math.PI / 180;
      return [opts.cx + opts.rx * Math.cos(r), opts.cy + opts.ry * Math.sin(r)];
    };

    // walk along the ellipse from a node centre until we leave its box (plus a margin)
    var clear = function (deg, dir) {
      var c = pt(deg), d = 0;
      while (d < step / 2) {
        var p = pt(deg + dir * d);
        if (Math.abs(p[0] - c[0]) > opts.w / 2 + 8 || Math.abs(p[1] - c[1]) > opts.h / 2 + 8) break;
        d += 0.5;
      }
      return deg + dir * d;
    };
    for (var i = 0; i < n; i++) {
      var a1 = pt(opts.gap ? start + i * step + gap : clear(start + i * step, 1)),
          a2 = pt(opts.gap ? start + (i + 1) * step - gap : clear(start + (i + 1) * step, -1));
      el('path', {
        d: 'M' + a1[0] + ',' + a1[1] + ' A' + opts.rx + ',' + opts.ry + ' 0 0 1 ' + a2[0] + ',' + a2[1],
        style: 'fill:none;stroke:var(--ink-3);stroke-width:2', 'marker-end': 'url(#' + arrowId + ')'
      }, svg);
    }

    if (opts.center) {
      opts.center.forEach(function (line, k) {
        var f = (opts.fs || 15) / 15;
        text(svg, opts.cx, opts.cy + f * (-(opts.center.length - 1) * 12 + k * 26 + 6), line,
          k === 0 ? 'font-size:' + 20 * f + 'px;font-weight:800;fill:var(--ink)' : 'font-size:' + 14 * f + 'px;fill:var(--ink-3)');
      });
    }

    opts.items.forEach(function (it, i) {
      var p = pt(start + i * step), g = el('g', {}, svg);
      el('rect', { x: p[0] - opts.w / 2, y: p[1] - opts.h / 2, width: opts.w, height: opts.h, rx: 12,
        style: 'fill:' + (it.fill || 'var(--brand-soft)') + ';stroke:' + (it.ink || 'var(--brand)') + ';stroke-width:1.5' }, g);
      var fs = opts.fs || 15;   // title font size; subtitle scales with it
      text(g, p[0], p[1] + (it.s ? -fs * 0.2 : fs * 0.4), (i + 1) + '. ' + it.t, 'font-size:' + fs + 'px;font-weight:700;fill:' + (it.ink || 'var(--brand-ink)'));
      if (it.s) text(g, p[0], p[1] + fs * 1.07, it.s, 'font-size:' + fs * 0.8 + 'px;fill:var(--ink-2)');
    });
    return { point: pt, arrowId: arrowId };
  };

  window.svgEl = el;
  window.svgText = text;
  window.svgArrow = ensureArrow;
})();
