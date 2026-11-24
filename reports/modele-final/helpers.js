  // ---------- helpers ----------
  function f(x, n) { return x == null ? "–" : Number(x).toFixed(n == null ? 3 : n); }
  function pc(x, n) { return x == null ? "–" : (100 * x).toFixed(n == null ? 1 : n) + "%"; }
  function h(tag, attrs, kids) {
    var e = document.createElement(tag);
    for (var k in (attrs || {})) { if (k === "text") e.textContent = attrs[k]; else if (k === "html") e.innerHTML = attrs[k]; else e.setAttribute(k, attrs[k]); }
    (kids || []).forEach(function (c) { if (c != null) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return e;
  }
  function s(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in (attrs || {})) { if (k === "text") e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (parent) parent.appendChild(e);
    return e;
  }
  function svgBox(w, hgt, label) {
    var svg = s("svg", { viewBox: "0 0 " + w + " " + hgt, role: "img", "aria-label": label });
    return svg;
  }
  function scale(d0, d1, r0, r1) { return function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); }; }
  function esc(t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  // ---------- tooltip ----------
  var tip = document.getElementById("tip");
  function showTip(ev, html) {
    tip.innerHTML = html; tip.hidden = false;
    var x = ev.clientX, y = ev.clientY;
    if (x == null && ev.target && ev.target.getBoundingClientRect) { var b = ev.target.getBoundingClientRect(); x = b.left + b.width / 2; y = b.top; }
    var tw = tip.offsetWidth, th = tip.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    var left = x + 14, top = y + 14;
    if (left + tw > vw - 8) left = x - tw - 14;
    if (top + th > vh - 8) top = y - th - 14;
    tip.style.left = Math.max(8, left) + "px"; tip.style.top = Math.max(8, top) + "px";
  }
  function hideTip() { tip.hidden = true; }
  function row(label, value, color) {
    return '<div class="tip-row">' + (color ? '<i class="tip-key" style="background:' + color + '"></i>' : "") +
      '<span class="tip-l">' + esc(label) + "</span><strong>" + value + "</strong></div>";
  }
  function hover(el, html) {
    el.setAttribute("tabindex", "0");
    el.addEventListener("pointermove", function (ev) { el.classList.add("is-hover"); showTip(ev, html()); });
    el.addEventListener("pointerleave", function () { el.classList.remove("is-hover"); hideTip(); });
    el.addEventListener("focus", function (ev) { el.classList.add("is-hover"); showTip(ev, html()); });
    el.addEventListener("blur", function () { el.classList.remove("is-hover"); hideTip(); });
  }

