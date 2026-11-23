(function () {
  "use strict";
  var $ = function (id) { return document.getElementById(id); };
  var MAX_KM = 20, MIN_KM = 0.5, NODATA = 65535, SCALE = 10000;
  var MAP_NAMES = ["Conservative", "Central", "Optimistic"];
  // canopy ramp, 0 -> 100% cover
  var RAMP = ["#f4f1e1", "#dfe9b8", "#b5d68f", "#7fbc72", "#4c9a5c", "#25744a", "#0d4b33"];
  var PRESETS = {
    example: { t: [0.2, 0.5, 0.8], names: ["Sparse", "Moderately dense", "Dense", "Very dense"] },
    project: { t: [0.1, 0.4, 0.7], names: ["Non-forest", "Sparse", "Moderate", "Dense"] },
    forest: { t: [0.1], names: ["Non-forest", "Forest"] },
    tenths: { t: [0.2, 0.4, 0.6, 0.8], names: ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"] }
  };
  var S = {
    aoi: null, rect: null, label: "", drawing: false, drawStart: null, preview: null,
    job: null, timer: null, res: null, resId: null, maps: null, ov: null,
    view: "side", colour: "continuous", scene: 0, overlay: null, lut: null, classes: null
  };

  // ---------- small helpers ----------
  function rgb(hex) { return [1, 3, 5].map(function (i) { return parseInt(hex.substr(i, 2), 16); }); }
  function toHex(c) { return "#" + c.map(function (v) { return Math.round(v).toString(16).padStart(2, "0"); }).join(""); }
  function rampAt(t) {
    var s = RAMP.map(rgb), x = Math.min(Math.max(t, 0), 1) * (s.length - 1), i = Math.min(s.length - 2, Math.floor(x)), u = x - i;
    return [0, 1, 2].map(function (k) { return s[i][k] + (s[i + 1][k] - s[i][k]) * u; });
  }
  function pct(v, d) { return v == null ? "no data" : (100 * v).toFixed(d == null ? 1 : d) + "%"; }
  function num(v) { return Math.round(v).toLocaleString("en-US"); }
  function esc(t) { return String(t).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function sizeKm(b) {
    var lat = (b[1] + b[3]) / 2 * Math.PI / 180, R = 6371;
    return [R * (b[2] - b[0]) * Math.PI / 180 * Math.cos(lat), R * (b[3] - b[1]) * Math.PI / 180];
  }
  function api(path, opts) {
    return fetch(path, opts).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (j) {
        if (!r.ok) throw new Error(j.error || ("Request failed (" + r.status + ")"));
        return j;
      });
    });
  }
  function save(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private mode */ } }
  function load(k) { try { return JSON.parse(localStorage.getItem(k)); } catch (e) { return null; } }
  function showError(msg) { var e = $("error"); e.textContent = msg; e.hidden = !msg; }

  // ---------- map ----------
  var map = L.map("map", { worldCopyJump: true }).setView([49.6, -84.5], 5);
  var bases = {
    map: L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
      { maxZoom: 19, attribution: "&copy; OpenStreetMap contributors" }),
    sat: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 19, attribution: "Imagery &copy; Esri, Maxar, Earthstar Geographics" })
  };
  bases.map.addTo(map);
  $("basemap").addEventListener("change", function (e) {
    Object.keys(bases).forEach(function (k) { map.removeLayer(bases[k]); });
    bases[e.target.value].addTo(map).bringToBack();
  });
  var AOI_STYLE = { color: "#1f5f3a", weight: 2, fillOpacity: 0.06 };

  function setDrawing(on) {
    S.drawing = on;
    $("draw-btn").setAttribute("aria-pressed", on ? "true" : "false");
    $("draw-btn").textContent = on ? "Cancel drawing" : "Draw an area";
    $("draw-hint").textContent = on ? "Click the first corner, then the opposite corner." : "Click two opposite corners on the map.";
    map.getContainer().style.cursor = on ? "crosshair" : "";
    if (!on) { S.drawStart = null; if (S.preview) { map.removeLayer(S.preview); S.preview = null; } }
  }
  $("draw-btn").addEventListener("click", function () { setDrawing(!S.drawing); });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") { setDrawing(false); $("search-results").hidden = true; } });
  map.on("click", function (e) {
    if (!S.drawing) return;
    if (!S.drawStart) {
      S.drawStart = e.latlng;
      S.preview = L.rectangle(L.latLngBounds(e.latlng, e.latlng), { color: "#1f5f3a", weight: 2, dashArray: "6 4", fillOpacity: 0.05 }).addTo(map);
    } else {
      var b = L.latLngBounds(S.drawStart, e.latlng);
      setDrawing(false);
      S.label = "";
      setAoi([b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], true);
    }
  });
  map.on("mousemove", function (e) {
    if (S.drawing && S.drawStart && S.preview) S.preview.setBounds(L.latLngBounds(S.drawStart, e.latlng));
  });

  function setAoi(b, fromMap) {
    b = b.map(function (v) { return Math.round(v * 1e5) / 1e5; });
    S.aoi = b;
    var bounds = [[b[1], b[0]], [b[3], b[2]]];
    if (S.rect) S.rect.setBounds(bounds); else S.rect = L.rectangle(bounds, AOI_STYLE).addTo(map);
    var corners = [[b[3], b[0]], [b[3], b[2]], [b[1], b[2]], [b[1], b[0]]];
    corners.forEach(function (c, i) { $("c" + i + "-lat").value = c[0].toFixed(5); $("c" + i + "-lon").value = c[1].toFixed(5); });
    var k = sizeKm(b), el = $("aoi-size");
    var cells = Math.round(k[0] * k[1] * 1e4);
    var bad = Math.max(k[0], k[1]) > MAX_KM ? "Too large: at most " + MAX_KM + " km on each side." :
      Math.min(k[0], k[1]) < MIN_KM ? "Too small: at least 500 m on each side." : "";
    el.textContent = k[0].toFixed(1) + " × " + k[1].toFixed(1) + " km (" + (k[0] * k[1]).toFixed(1) + " km², about " + num(cells) +
      " cells of 10 m)." + (bad ? " " + bad : "");
    el.classList.toggle("bad", !!bad);
    $("run-btn").disabled = !!bad;
    if (!fromMap) map.fitBounds(bounds, { padding: [30, 30], maxZoom: 14 });
  }

  $("corners-btn").addEventListener("click", function () {
    var lats = [], lons = [];
    for (var i = 0; i < 4; i++) {
      var la = parseFloat($("c" + i + "-lat").value.replace(",", ".")), lo = parseFloat($("c" + i + "-lon").value.replace(",", "."));
      if (!isFinite(la) || !isFinite(lo)) continue;
      lats.push(la); lons.push(lo);
    }
    if (lats.length < 2 || lats.some(function (v) { return Math.abs(v) > 85; }) || lons.some(function (v) { return Math.abs(v) > 180; })) {
      $("aoi-size").textContent = "Type at least two opposite corners, as decimal degrees (latitude between -85 and 85, longitude between -180 and 180).";
      $("aoi-size").classList.add("bad");
      return;
    }
    var b = [Math.min.apply(null, lons), Math.min.apply(null, lats), Math.max.apply(null, lons), Math.max.apply(null, lats)];
    if (b[0] === b[2] || b[1] === b[3]) { $("aoi-size").textContent = "The corners must span an area."; $("aoi-size").classList.add("bad"); return; }
    S.label = "";
    setAoi(b, false);
  });

  // ---------- place search ----------
  $("search-form").addEventListener("submit", function (e) {
    e.preventDefault();
    var q = $("search-q").value.trim(), list = $("search-results");
    if (!q) return;
    list.hidden = false;
    list.innerHTML = '<li class="hint" style="padding:8px">Searching…</li>';
    api("/api/geocode?q=" + encodeURIComponent(q)).then(function (rs) {
      list.innerHTML = "";
      if (!rs.length) { list.innerHTML = '<li class="hint" style="padding:8px">No place found. Try another spelling, or add the province.</li>'; return; }
      rs.forEach(function (r) {
        var b = document.createElement("button");
        b.type = "button"; b.textContent = r.name;
        b.addEventListener("click", function () { choosePlace(r); });
        var li = document.createElement("li"); li.appendChild(b); list.appendChild(li);
      });
    }).catch(function (err) { list.innerHTML = '<li class="error" style="padding:8px">' + esc(err.message) + "</li>"; });
  });
  document.addEventListener("click", function (e) {
    if (!$("search-results").contains(e.target) && !$("search-form").contains(e.target)) $("search-results").hidden = true;
  });
  function choosePlace(r) {
    $("search-results").hidden = true;
    $("search-q").value = r.name;
    S.label = r.name.split(",")[0];
    var b = null;
    if (r.bbox && r.bbox.length === 4) {
      var cand = [r.bbox[2], r.bbox[0], r.bbox[3], r.bbox[1]];   // Nominatim: south, north, west, east
      var k = sizeKm(cand);
      if (Math.max(k[0], k[1]) <= MAX_KM && Math.min(k[0], k[1]) >= 1) b = cand;
    }
    if (!b) {   // a point, or a large region: a 5 x 5 km square around the place
      var dLat = 2.5 / 111.2, dLon = 2.5 / (111.2 * Math.cos(r.lat * Math.PI / 180));
      b = [r.lon - dLon, r.lat - dLat, r.lon + dLon, r.lat + dLat];
    }
    setAoi(b, false);
  }

  // ---------- options ----------
  (function () {
    var sel = $("opt-year"), y = new Date().getFullYear();
    for (var k = y; k >= 2019; k--) { var o = document.createElement("option"); o.value = k; o.textContent = k; sel.appendChild(o); }
  })();

  // ---------- run and follow an analysis ----------
  $("run-btn").addEventListener("click", function () {
    if (!S.aoi) return;
    showError("");
    $("run-btn").disabled = true;
    api("/api/analyses", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bbox: S.aoi, max_scenes: +$("opt-scenes").value, year: $("opt-year").value || null, label: S.label })
    }).then(function (j) { watch(j.id); refreshHistory(); })
      .catch(function (err) { showError(err.message); $("run-btn").disabled = false; });
  });

  function watch(id) {
    S.job = id;
    clearInterval(S.timer);
    $("progress").hidden = false;
    $("run-btn").disabled = true;
    poll();
    S.timer = setInterval(poll, 1000);
  }
  function poll() {
    var id = S.job;
    api("/api/analyses/" + id).then(function (j) {
      if (id !== S.job) return;
      var msg = j.state === "queued" ? "Waiting: " + (j.queue_position || 1) + " analysis ahead" : j.message;
      $("progress-text").textContent = msg;
      $("progress-fill").style.width = Math.round(100 * (j.progress || 0)) + "%";
      if (j.state === "done" || j.state === "error") {
        clearInterval(S.timer);
        $("progress").hidden = true;
        $("run-btn").disabled = !S.aoi;
        refreshHistory();
        if (j.state === "error") showError(j.message); else openResult(j);
      }
    }).catch(function (err) { $("progress-text").textContent = "Connection problem, retrying (" + err.message + ")"; });
  }

  function refreshHistory() {
    api("/api/analyses").then(function (items) {
      var ul = $("history");
      ul.innerHTML = "";
      if (!items.length) { ul.innerHTML = '<li class="hint">None yet.</li>'; return; }
      items.slice(0, 12).forEach(function (j) {
        var b = document.createElement("button");
        var when = new Date(j.created * 1000).toLocaleString("en-CA", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
        var name = j.label || (j.bbox[1].toFixed(3) + ", " + j.bbox[0].toFixed(3));
        var st = j.state === "done" ? "summer " + j.year : j.state === "error" ? "failed" : j.state;
        b.innerHTML = esc(name) + ' <span class="st">· ' + esc(when) + " · " + esc(st) + "</span>";
        b.type = "button";
        b.addEventListener("click", function () {
          if (j.state === "done") api("/api/analyses/" + j.id).then(openResult);
          else if (j.state === "queued" || j.state === "running") watch(j.id);
          else showError(j.message);
        });
        var li = document.createElement("li"); li.appendChild(b); ul.appendChild(li);
        if ((j.state === "queued" || j.state === "running") && !S.job) watch(j.id);
      });
    }).catch(function () { /* the history is optional */ });
  }

  // ---------- results ----------
  function openResult(j) {
    var r = j.result;
    Promise.all(["maps.u16", "overlay.u16"].map(function (f) {
      return fetch("/api/analyses/" + j.id + "/" + f).then(function (x) { if (!x.ok) throw new Error("missing " + f); return x.arrayBuffer(); });
    })).then(function (bufs) {
      S.res = r; S.resId = j.id; S.maps = new Uint16Array(bufs[0]); S.ov = new Uint16Array(bufs[1]); S.scene = 0;
      S.label = j.label || "";
      setAoi(r.bbox, true);
      renderResult();
      $("results").hidden = false;
      $("results").scrollIntoView({ behavior: "smooth", block: "start" });
    }).catch(function (err) { showError("The maps could not be loaded: " + err.message); });
  }

  function renderResult() {
    var r = S.res;
    var dates = r.scenes.map(function (s) { return new Date(s.date + "T12:00:00Z").toLocaleDateString("en-CA", { day: "numeric", month: "short" }); });
    $("h-res").textContent = "Canopy cover" + (S.label ? ", " + S.label : "");
    $("res-meta").textContent = "Summer " + r.year + " · " + r.scenes.length + " image" + (r.scenes.length > 1 ? "s" : "") + " (" + dates.join(", ") + ") · " +
      r.size_km[0].toFixed(1) + " × " + r.size_km[1].toFixed(1) + " km · " + num(r.width * r.height) + " cells of 10 m · EPSG:" + r.epsg +
      " · computed in " + Math.round(r.seconds.total) + " s";
    var w = $("warnings");
    w.innerHTML = r.warnings.map(function (x) { return "<li>" + esc(x) + "</li>"; }).join("");
    w.hidden = !r.warnings.length;
    $("download").href = "/api/analyses/" + S.resId + "/canopy_cover.tif";
    var chips = $("scene-chips");
    chips.innerHTML = "";
    r.scenes.forEach(function (s, i) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = s.date; b.setAttribute("aria-pressed", i === S.scene ? "true" : "false");
      b.title = s.id;
      b.addEventListener("click", function () { S.scene = i; showScene(); });
      chips.appendChild(b);
    });
    showScene();
    document.querySelectorAll(".pnl canvas").forEach(function (c) { c.width = r.width; c.height = r.height; });
    $("overlay-on").disabled = false; $("overlay-opacity").disabled = false;
    applyView();
    redraw();
  }
  function showScene() {
    var s = S.res.scenes[S.scene];
    $("sat-img").src = "/api/analyses/" + S.resId + "/rgb_" + s.index + ".jpg";
    $("sat-date").textContent = s.date;
    document.querySelectorAll("#scene-chips button").forEach(function (b, i) { b.setAttribute("aria-pressed", i === S.scene ? "true" : "false"); });
  }

  // view and colour switches
  document.querySelectorAll("[data-view]").forEach(function (b) {
    b.addEventListener("click", function () { S.view = b.getAttribute("data-view"); applyView(); updateOverlay(); });
  });
  document.querySelectorAll("[data-colour]").forEach(function (b) {
    b.addEventListener("click", function () {
      S.colour = b.getAttribute("data-colour");
      document.querySelectorAll("[data-colour]").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); });
      $("class-editor").hidden = S.colour !== "classes";
      save("canopy-colour", S.colour);
      redraw();
    });
  });
  function applyView() {
    document.querySelectorAll("[data-view]").forEach(function (x) { x.setAttribute("aria-pressed", x.getAttribute("data-view") === S.view ? "true" : "false"); });
    var side = S.view === "side";
    $("panels").className = "panels " + (side ? "side" : "single");
    document.querySelectorAll(".pnl").forEach(function (p) {
      var k = p.getAttribute("data-k");
      p.hidden = !(k === "sat" || side || k === S.view);
    });
  }

  // ---------- colours ----------
  function buildLut() {
    var lut = new Uint8ClampedArray((SCALE + 1) * 4), c = S.classes;
    var cols = c.colors.map(rgb);
    for (var v = 0; v <= SCALE; v++) {
      var col;
      if (S.colour === "classes") col = cols[classOf(v / SCALE)];
      else col = rampAt(v / SCALE);
      lut[v * 4] = col[0]; lut[v * 4 + 1] = col[1]; lut[v * 4 + 2] = col[2]; lut[v * 4 + 3] = 255;
    }
    S.lut = lut;
  }
  function classOf(x) {
    var t = S.classes.t, i = 0;
    while (i < t.length && x >= t[i]) i++;
    return i;
  }
  function paint(canvas, arr, offset, w, h) {
    var ctx = canvas.getContext("2d"), img = ctx.createImageData(w, h), d = img.data, lut = S.lut;
    for (var i = 0, n = w * h; i < n; i++) {
      var v = arr[offset + i], j = i * 4;
      if (v === NODATA) { d[j + 3] = 0; continue; }
      var o = v * 4;
      d[j] = lut[o]; d[j + 1] = lut[o + 1]; d[j + 2] = lut[o + 2]; d[j + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }
  function redraw() {
    if (!S.res) return;
    buildLut();
    var r = S.res, n = r.width * r.height;
    document.querySelectorAll(".pnl[data-k] canvas").forEach(function (c) {
      var k = +c.closest(".pnl").getAttribute("data-k");
      paint(c, S.maps, k * n, r.width, r.height);
    });
    drawLegend();
    drawAreas();
    updateOverlay();
  }
  function drawLegend() {
    var el = $("legend");
    if (S.colour === "continuous") {
      var stops = RAMP.map(function (c, i) { return c + " " + Math.round(100 * i / (RAMP.length - 1)) + "%"; }).join(", ");
      el.innerHTML = 'Canopy cover 0%<span class="ramp" style="background:linear-gradient(90deg,' + stops + ')"></span>100%' +
        '<span><span class="swatch" style="background:repeating-linear-gradient(45deg,#e4e6df 0 3px,#d6d9d0 3px 6px)"></span>no data</span>';
    } else {
      var c = S.classes;
      el.innerHTML = c.names.map(function (nm, i) {
        var lo = i ? c.t[i - 1] : 0, hi = i < c.t.length ? c.t[i] : 1;
        return '<span><span class="swatch" style="background:' + c.colors[i] + '"></span>' + esc(nm) + " (" + Math.round(lo * 100) + "–" + Math.round(hi * 100) + "%)</span>";
      }).join("");
    }
  }
  function drawAreas() {
    var r = S.res, n = r.width * r.height, cellHa = (r.cell_m * r.cell_m) / 1e4, t = $("areas-table");
    var head = "<thead><tr><th></th>" + MAP_NAMES.map(function (m) { return "<th>" + m + "</th>"; }).join("") + "</tr></thead>";
    var body = "";
    if (S.colour === "continuous") {
      var sums = [0, 0, 0], valid = [0, 0, 0];
      for (var k = 0; k < 3; k++) for (var i = 0; i < n; i++) { var v = S.maps[k * n + i]; if (v !== NODATA) { sums[k] += v / SCALE; valid[k]++; } }
      body = "<tr><td>Mean canopy cover</td>" + sums.map(function (s, k) { return "<td>" + pct(s / Math.max(valid[k], 1)) + "</td>"; }).join("") + "</tr>" +
        "<tr><td>Area under canopy</td>" + sums.map(function (s) { return "<td>" + num(s * cellHa) + " ha</td>"; }).join("") + "</tr>" +
        "<tr><td>Area mapped</td>" + valid.map(function (v) { return "<td>" + num(v * cellHa) + " ha</td>"; }).join("") + "</tr>";
      $("areas-title").textContent = "Canopy area";
      $("areas-note").textContent = "Area under canopy: the sum over the cells of cover × 1 ha per 100 cells. The conservative and optimistic maps bracket it.";
    } else {
      var c = S.classes, counts = [0, 1, 2].map(function () { return new Array(c.names.length).fill(0); }), tot = [0, 0, 0];
      for (var k2 = 0; k2 < 3; k2++) for (var i2 = 0; i2 < n; i2++) {
        var v2 = S.maps[k2 * n + i2];
        if (v2 !== NODATA) { counts[k2][classOf(v2 / SCALE)]++; tot[k2]++; }
      }
      body = c.names.map(function (nm, ci) {
        return '<tr><td><span class="swatch" style="background:' + c.colors[ci] + '"></span>' + esc(nm) + "</td>" +
          [0, 1, 2].map(function (k) { return "<td>" + num(counts[k][ci] * cellHa) + " ha · " + pct(counts[k][ci] / Math.max(tot[k], 1), 0) + "</td>"; }).join("") + "</tr>";
      }).join("");
      $("areas-title").textContent = "Area by class";
      $("areas-note").textContent = "Each map classifies the same cells; the conservative map puts more of them in low-cover classes, the optimistic map in high-cover ones.";
    }
    t.innerHTML = head + "<tbody>" + body + "</tbody>";
  }

  // ---------- class editor ----------
  function defaultColours(n) {
    var out = [];
    for (var i = 0; i < n; i++) out.push(toHex(rampAt(n === 1 ? 0.5 : 0.08 + 0.92 * i / (n - 1))));
    return out;
  }
  function setPreset(key) {
    var p = PRESETS[key];
    S.classes = { t: p.t.slice(), names: p.names.slice(), colors: defaultColours(p.names.length) };
  }
  function renderEditor() {
    var c = S.classes, n = c.names.length, sel = $("ce-n");
    if (!sel.options.length) for (var k = 2; k <= 8; k++) { var o = document.createElement("option"); o.value = k; o.textContent = k; sel.appendChild(o); }
    sel.value = n;
    var rows = $("ce-rows");
    rows.innerHTML = "";
    c.names.forEach(function (nm, i) {
      var tr = document.createElement("tr");
      var from = i ? Math.round(c.t[i - 1] * 1000) / 10 + "%" : "0%";
      tr.innerHTML = '<td><input type="color" aria-label="Colour of class ' + (i + 1) + '" value="' + c.colors[i] + '"></td>' +
        '<td><input type="text" aria-label="Name of class ' + (i + 1) + '" value="' + esc(nm) + '"></td>' +
        '<td class="from">' + from + "</td>" +
        (i < n - 1 ? '<td><input type="number" aria-label="Upper bound of class ' + (i + 1) + ', percent" min="0.1" max="99.9" step="0.1" value="' +
          Math.round(c.t[i] * 1000) / 10 + '"> %</td>' : '<td class="from">100%</td>');
      var ins = tr.querySelectorAll("input");
      ins[0].addEventListener("input", function () { c.colors[i] = ins[0].value; changed(false); });
      ins[1].addEventListener("input", function () { c.names[i] = ins[1].value || "Class " + (i + 1); changed(false); });
      if (ins[2]) ins[2].addEventListener("change", function () {
        var v = parseFloat(ins[2].value) / 100, t = c.t.slice();
        t[i] = v;
        var ok = t.every(function (x, j) { return isFinite(x) && x > 0 && x < 1 && (j === 0 || x > t[j - 1]); });
        $("ce-error").hidden = ok;
        $("ce-error").textContent = ok ? "" : "Bounds must increase from one class to the next, between 0 and 100%.";
        if (ok) { c.t = t; changed(true); }
      });
      rows.appendChild(tr);
    });
  }
  function changed(rerenderEditor) {
    $("ce-preset").value = "";
    save("canopy-classes", S.classes);
    if (rerenderEditor) renderEditor();
    redraw();
  }
  $("ce-preset").addEventListener("change", function (e) {
    if (!e.target.value) return;
    setPreset(e.target.value);
    save("canopy-classes", S.classes);
    renderEditor(); redraw();
  });
  $("ce-n").addEventListener("change", function (e) {
    var n = +e.target.value, c = S.classes, t = [], names = [];
    for (var i = 1; i < n; i++) t.push(Math.round(1000 * i / n) / 1000);
    for (var j = 0; j < n; j++) names.push(c.names[j] || "Class " + (j + 1));
    S.classes = { t: t, names: names, colors: defaultColours(n) };
    $("ce-error").hidden = true;
    changed(true);
  });

  // ---------- reading values under the pointer ----------
  var marks = [];
  document.querySelectorAll(".pnl-img").forEach(function (box) {
    var m = document.createElement("div"); m.className = "mark"; m.hidden = true; box.appendChild(m); marks.push(m);
    box.addEventListener("pointermove", onMove);
    box.addEventListener("pointerdown", onMove);
    box.addEventListener("pointerleave", function () { marks.forEach(function (x) { x.hidden = true; }); });
  });
  function lonLat(u, v) {
    var c = S.res.corners;   // NW, NE, SE, SW as [lat, lon]
    var f = function (k) { return (1 - u) * (1 - v) * c[0][k] + u * (1 - v) * c[1][k] + u * v * c[2][k] + (1 - u) * v * c[3][k]; };
    return [f(0), f(1)];
  }
  function onMove(ev) {
    if (!S.res) return;
    var r = S.res, b = ev.currentTarget.getBoundingClientRect();
    var u = (ev.clientX - b.left) / b.width, v = (ev.clientY - b.top) / b.height;
    if (u < 0 || u >= 1 || v < 0 || v >= 1) return;
    var x = Math.floor(u * r.width), y = Math.floor(v * r.height), n = r.width * r.height;
    marks.forEach(function (m) { m.hidden = false; m.style.left = (u * 100) + "%"; m.style.top = (v * 100) + "%"; });
    var ll = lonLat((x + 0.5) / r.width, (y + 0.5) / r.height);
    var parts = ['<span class="lbl">' + Math.abs(ll[0]).toFixed(5) + "° " + (ll[0] >= 0 ? "N" : "S") + ", " +
      Math.abs(ll[1]).toFixed(5) + "° " + (ll[1] >= 0 ? "E" : "W") + "</span>"];
    for (var k = 0; k < 3; k++) {
      var val = S.maps[k * n + y * r.width + x], txt;
      if (val === NODATA) txt = "no data";
      else {
        txt = pct(val / SCALE);
        if (S.colour === "classes") { var ci = classOf(val / SCALE); txt += ' <span class="swatch" style="background:' + S.classes.colors[ci] + '"></span>' + esc(S.classes.names[ci]); }
      }
      parts.push('<span><span class="lbl">' + MAP_NAMES[k] + "</span> <b>" + txt + "</b></span>");
    }
    $("readout").innerHTML = parts.join("");
  }

  // ---------- result on the web map ----------
  function updateOverlay() {
    if (S.overlay) { map.removeLayer(S.overlay); S.overlay = null; }
    if (!$("overlay-on").checked || !S.ov || !S.res) return;
    var o = S.res.overlay, k = S.view === "side" ? 1 : +S.view;
    var c = document.createElement("canvas"); c.width = o.width; c.height = o.height;
    paint(c, S.ov, k * o.width * o.height, o.width, o.height);
    S.overlay = L.imageOverlay(c.toDataURL(), o.bounds, { opacity: $("overlay-opacity").value / 100, interactive: false }).addTo(map);
  }
  $("overlay-on").addEventListener("change", function () {
    updateOverlay();
    if ($("overlay-on").checked && S.res) map.fitBounds([[S.res.bbox[1], S.res.bbox[0]], [S.res.bbox[3], S.res.bbox[2]]], { padding: [30, 30] });
  });
  $("overlay-opacity").addEventListener("input", function () { if (S.overlay) S.overlay.setOpacity($("overlay-opacity").value / 100); });

  // ---------- start ----------
  S.classes = load("canopy-classes");
  if (!S.classes || !S.classes.names || !S.classes.t || S.classes.t.length !== S.classes.names.length - 1) setPreset("example");
  S.colour = load("canopy-colour") === "classes" ? "classes" : "continuous";
  document.querySelectorAll("[data-colour]").forEach(function (x) { x.setAttribute("aria-pressed", x.getAttribute("data-colour") === S.colour ? "true" : "false"); });
  $("class-editor").hidden = S.colour !== "classes";
  renderEditor();
  refreshHistory();
})();
