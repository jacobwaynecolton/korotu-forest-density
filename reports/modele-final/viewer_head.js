  // ---------- map viewer ----------
  var STOPS = ["#f6f9fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
  var LUT = (function () {
    var rgb = STOPS.map(function (c) { return [1, 3, 5].map(function (i) { return parseInt(c.substr(i, 2), 16); }); });
    var out = new Uint8ClampedArray(256 * 4);
    for (var v = 0; v < 256; v++) {
      if (v > 250) { out[v * 4 + 3] = 0; continue; }
      var t = v / 250 * (rgb.length - 1), i = Math.min(rgb.length - 2, Math.floor(t)), u = t - i;
      for (var k = 0; k < 3; k++) out[v * 4 + k] = Math.round(rgb[i][k] + (rgb[i + 1][k] - rgb[i][k]) * u);
      out[v * 4 + 3] = 255;
    }
    return out;
  })();
  var NAMES = ["Label", "Conservative", "Central", "Optimistic"];
  var state = { id: null, w: 0, h: 0, gray: null, token: 0 };
  var panels = Array.prototype.slice.call(document.querySelectorAll(".v-img"));
  var marks = panels.map(function (p) { var mk = h("div", { class: "v-mark", hidden: "" }); p.appendChild(mk); return mk; });
  var readout = document.getElementById("v-read");

