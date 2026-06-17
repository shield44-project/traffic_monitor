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
  let browserStream = null;
  let browserTimer = null;
  const liveData = [];
  const liveChart = lineChart("liveChart", [], [
    { label: "Congestion", data: liveData, borderColor: palette.red, backgroundColor: "rgba(239,68,68,.1)", fill: true }
  ]);

  document.getElementById("startLive")?.addEventListener("click", async () => {
    const form = document.getElementById("liveForm");
    const source = form.manualSource.value.trim() || form.source.value;
    errorBox?.classList.add("d-none");
    const result = await postJson("/api/live/start", { source });
    state.textContent = result.ok ? "Running" : "Error";
    state.className = result.ok ? "badge text-bg-success" : "badge text-bg-danger";
    if (!result.ok && errorBox) {
      errorBox.textContent = result.error || "Could not start live monitoring.";
      if (source === "0" || source === 0) {
        errorBox.textContent += " Hint: Server has no webcam. Use 'Start Browser Camera' instead.";
      }
      errorBox.classList.remove("d-none");
    }
  });

  document.getElementById("stopLive")?.addEventListener("click", async () => {
    await postJson("/api/live/stop");
    state.textContent = "Idle";
    state.className = "badge text-bg-secondary";
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
      browserVideo.classList.remove("d-none");
      liveFrame.classList.add("d-none");
      state.textContent = "Browser Camera";
      state.className = "badge text-bg-success";
      errorBox?.classList.add("d-none");
      
      // Ensure video is playing before starting timer
      browserVideo.onloadedmetadata = () => {
        if (browserTimer) clearInterval(browserTimer);
        browserTimer = setInterval(() => sendBrowserFrame(browserVideo), 2000);
      };
    } catch (error) {
      if (errorBox) {
        errorBox.textContent = `Camera Error: ${error.message}`;
        errorBox.classList.remove("d-none");
      }
    }
  });

  document.getElementById("stopBrowserCamera")?.addEventListener("click", async () => {
    if (browserTimer) clearInterval(browserTimer);
    browserTimer = null;
    if (browserStream) browserStream.getTracks().forEach((track) => track.stop());
    browserStream = null;
    browserVideo.classList.add("d-none");
    liveFrame.classList.remove("d-none");
    state.textContent = "Idle";
    state.className = "badge text-bg-secondary";
    await postJson("/api/live/stop");
  });

  let lastTimestamp = null;
  setInterval(async () => {
    const response = await fetch("/api/live/status");
    const data = await response.json();
    if (!data.ok) return;
    
    const payload = data.payload || {};
    if (!payload.congestion) {
      state.textContent = data.running ? "Running (Initializing)" : "Idle";
      state.className = data.running ? "badge text-bg-info" : "badge text-bg-secondary";
      return;
    }

    state.textContent = data.running ? "Running" : "Idle";
    state.className = data.running ? "badge text-bg-success" : "badge text-bg-secondary";
    
    if (data.error && errorBox) {
      errorBox.textContent = data.error;
      errorBox.classList.remove("d-none");
    }

    if (payload.timestamp === lastTimestamp) return;
    lastTimestamp = payload.timestamp;

    document.getElementById("snapshotTime").textContent = payload.timestamp || "--";
    document.getElementById("liveVehicles").textContent = payload.congestion.total_count;
    document.getElementById("liveDensity").textContent = `${payload.congestion.density}%`;
    document.getElementById("liveCongestion").textContent = payload.congestion.level;
    document.getElementById("liveCo2").textContent = payload.emissions.co2e || payload.emissions.co2;
    
    if (payload.emergency_model_status && payload.emergency_model_status !== "ready" && errorBox) {
      errorBox.textContent = `Model Status: ${payload.emergency_model_status}`;
      errorBox.classList.remove("d-none");
    }
    
    alertBox.classList.toggle("d-none", !payload.emergency || payload.emergency.length === 0);
    
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
        alert(data.error || "Analysis failed");
      }
    } catch (e) {
      alert("Network error: " + e.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalText;
    }
  });
}

function showVideoResult(res) {
  document.getElementById("vResCongestion").textContent = res.avg_congestion_score;
  document.getElementById("vResDensity").textContent = `${res.avg_density}%`;
  document.getElementById("vResEmissions").textContent = res.emission_totals.co2e || res.emission_totals.co2;

  const tbody = document.getElementById("vResTable");
  tbody.innerHTML = Object.entries(res.vehicle_totals).map(([type, count]) => {
    // Basic estimation for NOx based on count for the report
    const co2 = (res.emission_totals.co2 / res.frames_analyzed * count).toFixed(1);
    return `<tr><td>${type}</td><td>${count}</td><td>${co2}</td><td>--</td></tr>`;
  }).join("");

  const previews = document.getElementById("vResPreviews");
  previews.innerHTML = (res.preview_frames || []).map(src => 
    `<img src="${src}" class="rounded border border-secondary" style="height: 100px; width: auto;">`
  ).join("");

  const modal = new bootstrap.Modal(document.getElementById("videoResultModal"));
  modal.show();
}

async function sendBrowserFrame(video) {
  if (!video || !video.videoWidth || !video.videoHeight) return;
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async (blob) => {
    if (!blob) return;
    const formData = new FormData();
    formData.append("frame", blob, "browser-frame.jpg");
    const response = await fetch("/api/analyze-browser-frame", { method: "POST", body: formData });
    const data = await response.json();
    if (data.ok && data.result) {
      document.getElementById("snapshotTime").textContent = data.result.timestamp || "--";
      document.getElementById("liveVehicles").textContent = data.result.congestion.total_count;
      document.getElementById("liveDensity").textContent = `${data.result.congestion.density}%`;
      document.getElementById("liveCongestion").textContent = data.result.congestion.level;
      document.getElementById("liveCo2").textContent = data.result.emissions.co2e || data.result.emissions.co2;
      document.getElementById("liveAlert").classList.toggle("d-none", !data.result.emergency || data.result.emergency.length === 0);
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
