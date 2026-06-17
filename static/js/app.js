(function () {
  const savedTheme = localStorage.getItem("traffic-theme");
  if (savedTheme) document.documentElement.setAttribute("data-bs-theme", savedTheme);
  document.addEventListener("click", (event) => {
    const button = event.target.closest("#themeToggle");
    if (!button) return;
    const current = document.documentElement.getAttribute("data-bs-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-bs-theme", next);
    localStorage.setItem("traffic-theme", next);
  });
})();

const palette = {
  green: "#22c55e",
  blue: "#38bdf8",
  red: "#ef4444",
  amber: "#f59e0b",
  slate: "#94a3b8",
  violet: "#a78bfa",
  cyan: "#06b6d4",
  pink: "#ec4899"
};

function labels(rows) {
  return rows.map((row) => (row.timestamp || row.hour_bucket || "").slice(5, 16));
}

function lineChart(canvasId, chartLabels, datasets) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  return new Chart(canvas, {
    type: "line",
    data: { labels: chartLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      tension: 0.32,
      plugins: { legend: { labels: { boxWidth: 10 } } },
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: window.innerWidth < 640 ? 6 : 12 } },
        y: { beginAtZero: true, suggestedMax: 100 }
      }
    }
  });
}

function doughnutChart(canvasId, chartLabels, data, colors) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  return new Chart(canvas, {
    type: "doughnut",
    data: { labels: chartLabels, datasets: [{ data, backgroundColor: colors, borderWidth: 0 }] },
    options: { responsive: true, maintainAspectRatio: false, cutout: "62%" }
  });
}

function barChart(canvasId, chartLabels, datasets, stacked = false) {
  const canvas = document.getElementById(canvasId);
  if (!canvas || !window.Chart) return null;
  return new Chart(canvas, {
    type: "bar",
    data: { labels: chartLabels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 10 } } },
      scales: {
        x: { stacked, grid: { display: false }, ticks: { maxRotation: window.innerWidth < 640 ? 0 : 50 } },
        y: { stacked, beginAtZero: true }
      }
    }
  });
}

function initDashboardCharts(initial) {
  const traffic = initial.traffic || [];
  lineChart("trafficChart", labels(traffic), [
    { label: "Density", data: traffic.map((r) => r.density), borderColor: palette.blue, backgroundColor: "rgba(56,189,248,.12)", fill: true },
    { label: "Score", data: traffic.map((r) => r.congestion_score), borderColor: palette.green, backgroundColor: "rgba(34,197,94,.08)", fill: true }
  ]);
  const totals = initial.vehicleTotals || {};
  doughnutChart("vehicleChart", Object.keys(totals), Object.values(totals), [palette.green, palette.blue, palette.amber, palette.red, palette.violet]);
}

function initHourlyTraffic(rows) {
  const ordered = rows || [];
  lineChart("hourlyTrafficChart", ordered.map((r) => r.hour_bucket.slice(5, 16)), [
    { label: "Avg Density", data: ordered.map((r) => r.avg_density), borderColor: palette.blue, backgroundColor: "rgba(56,189,248,.12)", fill: true },
    { label: "Avg Congestion", data: ordered.map((r) => r.avg_congestion), borderColor: palette.red, backgroundColor: "rgba(239,68,68,.1)", fill: true }
  ]);
}

function latestGasRisk(rows, summary) {
  const latest = rows && rows.length ? rows[rows.length - 1] : {};
  if (latest.gas_risk) return latest.gas_risk;
  return {
    co2e: Math.min(100, (summary.avg_co2e || summary.avg_co2 || 0) / 90),
    nox: Math.min(100, (summary.avg_nox || 0) * 2.8),
    pm25: Math.min(100, (summary.avg_pm25 || 0) * 65),
    pm10: Math.min(100, (summary.avg_pm10 || 0) * 40),
    co: Math.min(100, (summary.avg_co || 0) * 2.8),
    voc: Math.min(100, (summary.avg_voc || 0) * 25)
  };
}

function initEmissionCharts(rows, summary, vehicleEmissions, pollutantHealth) {
  const recent = rows || [];
  lineChart("climateEmissionChart", labels(recent), [
    { label: "CO2e", data: recent.map((r) => r.co2e || r.co2 || 0), borderColor: palette.green, backgroundColor: "rgba(34,197,94,.10)", fill: true },
    { label: "CO2", data: recent.map((r) => r.co2 || 0), borderColor: palette.blue, backgroundColor: "rgba(56,189,248,.08)", fill: false }
  ]);
  lineChart("airQualityEmissionChart", labels(recent), [
    { label: "NOx", data: recent.map((r) => r.nox || 0), borderColor: palette.amber, backgroundColor: "rgba(245,158,11,.10)", fill: true },
    { label: "CO", data: recent.map((r) => r.co || 0), borderColor: palette.violet, backgroundColor: "rgba(167,139,250,.08)", fill: true },
    { label: "PM2.5", data: recent.map((r) => r.pm25 || 0), borderColor: palette.red, backgroundColor: "rgba(239,68,68,.10)", fill: true },
    { label: "PM10", data: recent.map((r) => r.pm10 || 0), borderColor: palette.pink, backgroundColor: "rgba(236,72,153,.08)", fill: true }
  ]);

  const vehicles = Object.keys(vehicleEmissions || {});
  barChart("vehicleEmissionChart", vehicles, [
    { label: "CO2e", data: vehicles.map((v) => vehicleEmissions[v].co2e || 0), backgroundColor: palette.green },
    { label: "NOx x100", data: vehicles.map((v) => (vehicleEmissions[v].nox || 0) * 100), backgroundColor: palette.amber },
    { label: "PM2.5 x10000", data: vehicles.map((v) => (vehicleEmissions[v].pm25 || 0) * 10000), backgroundColor: palette.red }
  ]);

  const risk = latestGasRisk(recent, summary || {});
  const riskKeys = ["pm25", "pm10", "nox", "co", "voc", "co2e"];
  barChart("gasRiskChart", riskKeys.map((k) => k.toUpperCase()), [
    { label: "Risk score", data: riskKeys.map((k) => risk[k] || 0), backgroundColor: [palette.red, palette.pink, palette.amber, palette.violet, palette.cyan, palette.green] }
  ]);
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined
  });
  return response.json();
}

function initLivePage() {
  const state = document.getElementById("liveState");
  const alertBox = document.getElementById("liveAlert");
  const errorBox = document.getElementById("liveError");
  const browserVideo = document.getElementById("browserCamera");
  const liveFrame = document.getElementById("liveFrame");
  const placeholder = document.getElementById("cameraPlaceholder");
  let browserStream = null;
  let browserTimer = null;
  let browserFrameCount = 0;
  const liveData = [];
  const liveChart = lineChart("liveChart", [], [
    { label: "Congestion", data: liveData, borderColor: palette.red, backgroundColor: "rgba(239,68,68,.1)", fill: true }
  ]);

  function setStatus(text, className) {
    if (!state) return;
    state.textContent = text;
    state.className = className;
  }

  function showError(message) {
    if (!errorBox) return;
    errorBox.textContent = message;
    errorBox.classList.remove("d-none");
  }

  function setBrowserHealth(status, count, analyzedAt) {
    const statusEl = document.getElementById("browserCameraState");
    const countEl = document.getElementById("browserFrameCount");
    const analyzedEl = document.getElementById("browserLastAnalyzed");
    if (statusEl) statusEl.textContent = status;
    if (countEl && count !== undefined) countEl.textContent = count;
    if (analyzedEl && analyzedAt !== undefined) analyzedEl.textContent = analyzedAt;
  }

  function showAnnotatedBrowserFrame(payload) {
    if (!payload.preview_data_url || !liveFrame) return;
    liveFrame.src = payload.preview_data_url;
    browserVideo?.classList.add("capture-only");
    setVideoMode("server");
  }

  function startBrowserFrameLoop() {
    if (browserTimer) clearInterval(browserTimer);
    setBrowserHealth("Camera preview on; analyzing frames...", browserFrameCount, undefined);
    sendBrowserFrame(browserVideo, {
      onStart: () => setBrowserHealth("Sending frame to model...", browserFrameCount, undefined),
      onResult: (payload) => {
        browserFrameCount += 1;
        setBrowserHealth("Analyzing live frames", browserFrameCount, (payload.timestamp || "").slice(11, 19) || "now");
        showAnnotatedBrowserFrame(payload);
        updateSnapshot(payload);
      },
      onError: (message) => setBrowserHealth(message, browserFrameCount, undefined)
    });
    browserTimer = setInterval(() => {
      sendBrowserFrame(browserVideo, {
        onStart: () => setBrowserHealth("Sending frame to model...", browserFrameCount, undefined),
        onResult: (payload) => {
          browserFrameCount += 1;
          setBrowserHealth("Analyzing live frames", browserFrameCount, (payload.timestamp || "").slice(11, 19) || "now");
          showAnnotatedBrowserFrame(payload);
          updateSnapshot(payload);
        },
        onError: (message) => setBrowserHealth(message, browserFrameCount, undefined)
      });
    }, 2000);
  }

  function setVideoMode(mode) {
    placeholder?.classList.toggle("d-none", mode !== "empty");
    liveFrame?.classList.toggle("d-none", mode === "empty");
    browserVideo?.classList.toggle("d-none", mode !== "browser");
  }

  function updateSnapshot(payload) {
    if (!payload || !payload.congestion) return;
    document.getElementById("snapshotTime").textContent = payload.timestamp || "--";
    document.getElementById("liveVehicles").textContent = payload.congestion.total_count;
    document.getElementById("liveDensity").textContent = `${payload.congestion.density}%`;
    document.getElementById("liveCongestion").textContent = payload.congestion.level;
    const co2e = payload.emissions.co2e ?? payload.emissions.co2 ?? 0;
    document.getElementById("liveCo2").textContent = `${Number(co2e).toFixed(2)} g/km-eq`;
    const basis = document.getElementById("liveEmissionBasis");
    if (basis && payload.emission_basis) {
      basis.textContent = `CO2e = ${payload.emission_basis.co2e_formula}; ${payload.emission_basis.scope}.`;
    }
    const roi = payload.vehicles?.roi_density || {};
    const left = roi.left_lane || payload.impact?.left_lane || {};
    const right = roi.right_lane || payload.impact?.right_lane || {};
    document.getElementById("leftLaneCount").textContent = left.count ?? 0;
    document.getElementById("leftLaneState").textContent = left.intensity || "Smooth";
    document.getElementById("rightLaneCount").textContent = right.count ?? 0;
    document.getElementById("rightLaneState").textContent = right.intensity || "Smooth";
    document.getElementById("fuelWaste").textContent = `${payload.impact?.fuel_waste_l ?? 0} L`;
    document.getElementById("emergencyCount").textContent = payload.emergency?.length || 0;
    alertBox?.classList.toggle("d-none", !payload.emergency || payload.emergency.length === 0);
  }

  setVideoMode("empty");

  document.getElementById("startLive")?.addEventListener("click", async () => {
    const form = document.getElementById("liveForm");
    const source = form.manualSource.value.trim() || form.source.value;
    errorBox?.classList.add("d-none");
    const result = await postJson("/api/live/start", { source });
    setStatus(result.ok ? "Running" : "Error", result.ok ? "badge text-bg-success" : "badge text-bg-danger");
    if (result.ok) {
      if (liveFrame?.dataset.streamUrl) {
        liveFrame.src = `${liveFrame.dataset.streamUrl}?t=${Date.now()}`;
      }
      browserVideo?.classList.remove("capture-only");
      setVideoMode("server");
    }
    if (!result.ok && errorBox) {
      let message = result.error || "Could not start live monitoring.";
      if (source === "0" || source === 0) {
        message += " Hint: server webcam means a camera attached to this machine. Use phone/browser camera for your own device.";
      }
      showError(message);
    }
  });

  document.getElementById("stopLive")?.addEventListener("click", async () => {
    await postJson("/api/live/stop");
    setStatus("Idle", "badge text-bg-secondary");
    setVideoMode("empty");
  });

  document.getElementById("startBrowserCamera")?.addEventListener("click", async () => {
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Browser camera access requires HTTPS or localhost.");
      }
      browserStream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: "environment" },
        audio: false 
      });
      browserVideo.srcObject = browserStream;
      await browserVideo.play();
      browserVideo.classList.remove("d-none");
      setVideoMode("browser");
      setStatus("Browser Camera", "badge text-bg-success");
      browserFrameCount = 0;
      setBrowserHealth("Camera preview on; waiting for model...", browserFrameCount, "--");
      errorBox?.classList.add("d-none");

      if (browserVideo.readyState >= 2) {
        startBrowserFrameLoop();
      } else {
        browserVideo.onloadedmetadata = startBrowserFrameLoop;
        browserVideo.oncanplay = startBrowserFrameLoop;
      }
    } catch (error) {
      if (errorBox) {
        errorBox.textContent = `Camera Error: ${error.message}`;
        errorBox.classList.remove("d-none");
      }
      setBrowserHealth(`Camera error: ${error.message}`, browserFrameCount, undefined);
    }
  });

  document.getElementById("stopBrowserCamera")?.addEventListener("click", async () => {
    if (browserTimer) clearInterval(browserTimer);
    browserTimer = null;
    if (browserStream) browserStream.getTracks().forEach((track) => track.stop());
    browserStream = null;
    browserVideo?.classList.remove("capture-only");
    window.__browserFrameInFlight = false;
    setVideoMode("empty");
    setStatus("Idle", "badge text-bg-secondary");
    setBrowserHealth("Stopped", browserFrameCount, undefined);
    await postJson("/api/live/stop");
  });

  let lastTimestamp = null;
  setInterval(async () => {
    const response = await fetch("/api/live/status");
    const data = await response.json();
    if (!data.ok) return;
    
    const payload = data.payload || {};
    if (!payload.congestion) {
      setStatus(data.running ? "Running (Initializing)" : "Idle", data.running ? "badge text-bg-info" : "badge text-bg-secondary");
      return;
    }

    setStatus(data.running ? (data.mode === "browser" ? "Browser Camera" : "Running") : "Idle", data.running ? "badge text-bg-success" : "badge text-bg-secondary");
    if (data.running && data.mode === "server") setVideoMode("server");
    
    if (data.error && errorBox) {
      showError(data.error);
    }

    if (payload.timestamp === lastTimestamp) return;
    lastTimestamp = payload.timestamp;

    updateSnapshot(payload);
    
    if (payload.emergency_model_status && payload.emergency_model_status !== "ready" && errorBox) {
      showError(`Model Status: ${payload.emergency_model_status}`);
    }

    if (liveChart) {
      liveChart.data.labels.push((payload.timestamp || "").slice(11, 19));
      liveChart.data.datasets[0].data.push(payload.congestion.congestion_score);
      if (liveChart.data.labels.length > 25) {
        liveChart.data.labels.shift();
        liveChart.data.datasets[0].data.shift();
      }
      liveChart.update("none");
    }
  }, 1500);

  document.getElementById("imageAnalyzeForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const output = document.getElementById("imageResult");
    output.classList.remove("d-none");
    output.textContent = "Analyzing...";
    const formData = new FormData(event.target);
    const response = await fetch("/api/analyze-image", { method: "POST", body: formData });
    const result = await response.json();
    output.textContent = JSON.stringify(result, null, 2);
    if (result.ok) updateSnapshot(result.result);
  });

  document.getElementById("videoAnalyzeForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const btn = event.target.querySelector("button[type='submit']");
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Processing...';
    
    try {
      const formData = new FormData(event.target);
      const response = await fetch("/api/analyze-video", { method: "POST", body: formData });
      const data = await response.json();
      if (data.ok && data.result) {
        showVideoResult(data.result);
      } else {
        showError(data.error || "Analysis failed");
      }
    } catch (e) {
      showError("Network error: " + e.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalText;
    }
  });

}

function showVideoResult(res) {
  const totalUnique = Number(res.unique_vehicle_count || Object.values(res.vehicle_totals || {}).reduce((sum, count) => sum + Number(count || 0), 0));
  const avgVisible = Object.values(res.avg_visible_counts || {}).reduce((sum, count) => sum + Number(count || 0), 0);
  document.getElementById("vResUniqueVehicles").textContent = totalUnique;
  document.getElementById("vResCongestion").textContent = res.avg_congestion_score;
  document.getElementById("vResDensity").textContent = `${res.avg_density}%`;
  document.getElementById("vResSpeed").textContent = `${res.avg_speed || 0} km/h`;
  const totalCo2e = Number(res.emission_totals.co2e || res.emission_totals.co2 || 0);
  document.getElementById("vResEmissions").textContent = `${totalCo2e.toFixed(2)} g/km-eq`;
  const avgVisibleEl = document.getElementById("vResAvgVisible");
  if (avgVisibleEl) {
    avgVisibleEl.textContent = avgVisible.toFixed(2);
  }
  const basisEl = document.getElementById("vResBasis");
  if (basisEl) {
    basisEl.textContent = `${res.counting_basis || "Vehicle totals are unique tracked objects."} ${res.tracking?.note || ""} CO2e scope: ${res.emission_basis?.scope || "estimated g/km-equivalent"}.`;
  }
  document.getElementById("vResFuel").textContent = `${res.fuel_waste_l || 0} L`;
  document.getElementById("vResPeak").textContent = `Frame ${res.peak_frame?.frame ?? "--"} (${res.peak_frame?.total ?? 0} vehicles)`;
  document.getElementById("vResLanes").textContent = `${res.roi_density?.avg_left_lane || 0} / ${res.roi_density?.avg_right_lane || 0}`;

  const tbody = document.getElementById("vResTable");
  const total = totalUnique;
  tbody.innerHTML = Object.entries(res.vehicle_totals || {}).map(([type, count]) => {
    const share = total ? `${Math.round((Number(count) / total) * 100)}%` : "0%";
    const co2e = res.type_emission_totals?.[type]?.co2e || "--";
    return `<tr><td>${type}</td><td>${count}</td><td>${share}</td><td>${co2e} g/km-eq</td></tr>`;
  }).join("");

  const gasBox = document.getElementById("vResGases");
  if (gasBox && res.emission_totals) {
    const skip = ["co2e", "co2", "emission_score"];
    // List of harmful gases to emphasize
    const harmfulGases = {
      "nox": "Nitrogen Oxides (Toxic)",
      "pm25": "PM2.5 (Fine Dust)",
      "pm10": "PM10 (Coarse Dust)",
      "co": "Carbon Monoxide (Poisonous)",
      "voc": "VOCs (Smog precursor)",
      "so2": "Sulfur Dioxide (Irritant)",
      "ch4": "Methane (Greenhouse)",
      "n2o": "Nitrous Oxide (Greenhouse)",
      "hc": "Hydrocarbons"
    };

    gasBox.innerHTML = Object.entries(res.emission_totals)
      .filter(([k]) => !skip.includes(k) && typeof res.emission_totals[k] === "number")
      .map(([k, v]) => `<div><span>${harmfulGases[k] || k.toUpperCase()}</span><strong>${v.toFixed(4)} g</strong></div>`)
      .join("");
  }

  const previews = document.getElementById("vResPreviews");
  previews.innerHTML = (res.preview_frames || []).map(src => 
    `<img src="${src}" alt="Analyzed preview frame" style="cursor: pointer;" onclick="openPreview('${src}')">`
  ).join("");


  const timelineCanvas = document.getElementById("videoTimelineChart");
  if (timelineCanvas && window.Chart) {
    if (window.vTimelineChartInstance && typeof window.vTimelineChartInstance.destroy === "function") {
      window.vTimelineChartInstance.destroy();
    }
    window.vTimelineChartInstance = new Chart(timelineCanvas, {
      type: "line",
      data: {
        labels: (res.timeline || []).map((row) => row.frame),
        datasets: [
          { label: "Vehicles", data: (res.timeline || []).map((row) => row.total), borderColor: palette.green, backgroundColor: "rgba(34,197,94,.12)", fill: true },
          { label: "CO2e", data: (res.timeline || []).map((row) => row.co2e), borderColor: palette.amber, backgroundColor: "rgba(245,158,11,.08)", fill: false }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { boxWidth: 10 } } },
        scales: { x: { grid: { display: false } }, y: { beginAtZero: true } }
      }
    });
  }

  const modalEl = document.getElementById("videoResultModal");
  if (modalEl) {
    const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
    modal.show();
  }
}

function openPreview(src) {
  const modalImg = document.getElementById("previewModalImage");
  if (modalImg) {
    modalImg.src = src;
    const modal = new bootstrap.Modal(document.getElementById("previewImageModal"));
    modal.show();
  }
}

async function sendBrowserFrame(video, hooks = {}) {
  if (window.__browserFrameInFlight) return;
  if (!video || !video.videoWidth || !video.videoHeight) {
    hooks.onError?.("Camera preview active; waiting for video size");
    return;
  }
  window.__browserFrameInFlight = true;
  hooks.onStart?.();
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async (blob) => {
    try {
      if (!blob) {
        hooks.onError?.("Could not capture browser frame");
        window.__browserFrameInFlight = false;
        return;
      }
      const formData = new FormData();
      formData.append("frame", blob, "browser-frame.jpg");
      const response = await fetch("/api/analyze-browser-frame", { method: "POST", body: formData });
      const data = await response.json();
      if (data.ok && data.result) {
        hooks.onResult?.(data.result);
      } else {
        hooks.onError?.(data.error || "Model analysis failed");
      }
    } catch (error) {
      hooks.onError?.(`Frame upload failed: ${error.message}`);
    } finally {
      window.__browserFrameInFlight = false;
    }
  }, "image/jpeg", 0.82);
}

document.addEventListener("click", async (event) => {
  if (!event.target.closest("#runForecast")) return;
  const response = await postJson("/api/predict/traffic");
  const tbody = document.getElementById("predictionRows");
  if (!tbody || !response.ok) return;
  tbody.innerHTML = response.predictions.map((row) => (
    `<tr><td>now</td><td>${row.horizon_min} min</td><td>${row.future_congestion}</td><td><span class="level-pill">${row.future_level}</span></td></tr>`
  )).join("");
});

document.addEventListener("click", async (event) => {
  if (!event.target.closest("#refreshSummary")) return;
  const response = await fetch("/api/summary");
  const data = await response.json();
  const summary = data.summary || {};
  document.getElementById("metricVehicles") && (document.getElementById("metricVehicles").textContent = summary.total_vehicles);
  document.getElementById("metricEmergencies") && (document.getElementById("metricEmergencies").textContent = summary.total_emergencies);
  if (summary.current_congestion && document.getElementById("metricCongestion")) {
    document.getElementById("metricCongestion").textContent = summary.current_congestion.congestion_level;
  }
  if (summary.current_emission && document.getElementById("metricEmission")) {
    document.getElementById("metricEmission").textContent = summary.current_emission.category;
  }
});
