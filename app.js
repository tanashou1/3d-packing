const canvas = document.querySelector("#viewer");
const gl = canvas.getContext("webgl", { antialias: true });
const select = document.querySelector("#result-select");
const stats = document.querySelector("#stats");
const statusEl = document.querySelector("#status");
const attributionEl = document.querySelector("#attribution");

let program;
let buffers;
let vertexCount = 0;
let rotationX = -0.65;
let rotationY = 0.75;
let zoom = 2.65;
let dragging = false;
let lastMouse = [0, 0];

if (!gl) {
  statusEl.textContent = "WebGL is not available in this browser.";
} else {
  init();
}

async function init() {
  program = createProgram();
  buffers = {
    position: gl.createBuffer(),
    normal: gl.createBuffer(),
  };
  installControls();
  const results = await fetch("assets/results.json").then((r) => r.json());
  for (const result of results) {
    const option = document.createElement("option");
    option.value = result.id;
    option.textContent = result.title;
    select.appendChild(option);
  }
  select.addEventListener("change", () => {
    loadResult(results.find((result) => result.id === select.value));
  });
  await loadResult(results[0]);
  requestAnimationFrame(draw);
}

async function loadResult(result) {
  select.value = result.id;
  statusEl.textContent = `Loading ${result.title}...`;
  stats.innerHTML = [
    ["Packed objects", result.packedObjects],
    result.inputObjects ? ["Input objects", result.inputObjects] : null,
    ["Tray", result.tray],
    ["Voxel", result.voxel],
    ["Voxel density", `${result.voxelDensity}%`],
    ["Mesh density", `${result.meshDensity}%`],
    result.requiresStackingByBbox !== undefined
      ? ["BBox stacking required", result.requiresStackingByBbox ? "yes" : "no"]
      : null,
    result.sumBboxFootprint !== undefined
      ? ["Sum bbox footprint", result.sumBboxFootprint]
      : null,
    result.trayFootprint !== undefined ? ["Tray footprint", result.trayFootprint] : null,
    result.sumBboxVolume !== undefined ? ["Sum bbox volume", result.sumBboxVolume] : null,
    result.trayVolume !== undefined ? ["Tray volume", result.trayVolume] : null,
    ["Ray disassembly", result.rayDisassembly],
    ["Source", result.source],
  ]
    .filter(Boolean)
    .map(([key, value]) => `<div class="stat"><span>${key}</span><strong>${value}</strong></div>`)
    .join("");

  attributionEl.textContent = result.attribution
    ? "Loading attribution..."
    : result.source;

  const stlText = await fetch(result.stl).then((r) => {
    if (!r.ok) {
      throw new Error(`Failed to load ${result.stl}: ${r.status}`);
    }
    return r.text();
  });
  const mesh = parseAsciiStl(stlText);
  uploadMesh(mesh);

  if (result.attribution) {
    const attribution = await fetch(result.attribution).then((r) => r.json());
    attributionEl.textContent = JSON.stringify(attribution, null, 2);
  }
  statusEl.textContent = `${result.title}: ${mesh.triangles} triangles`;
}

function parseAsciiStl(text) {
  const positions = [];
  const normals = [];
  const vertices = [];
  let currentNormal = [0, 0, 1];
  let min = [Infinity, Infinity, Infinity];
  let max = [-Infinity, -Infinity, -Infinity];

  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    const parts = line.split(/\s+/);
    if (parts[0] === "facet" && parts[1] === "normal") {
      currentNormal = parts.slice(2, 5).map(Number);
    } else if (parts[0] === "vertex") {
      const vertex = parts.slice(1, 4).map(Number);
      vertices.push(vertex);
      for (let axis = 0; axis < 3; axis++) {
        min[axis] = Math.min(min[axis], vertex[axis]);
        max[axis] = Math.max(max[axis], vertex[axis]);
      }
      positions.push(...vertex);
      normals.push(...currentNormal);
    }
  }

  if (vertices.length === 0) {
    throw new Error("No vertices found in STL.");
  }

  const center = [
    (min[0] + max[0]) * 0.5,
    (min[1] + max[1]) * 0.5,
    (min[2] + max[2]) * 0.5,
  ];
  const maxDim = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
  for (let i = 0; i < positions.length; i += 3) {
    positions[i] = (positions[i] - center[0]) / maxDim;
    positions[i + 1] = (positions[i + 1] - center[1]) / maxDim;
    positions[i + 2] = (positions[i + 2] - center[2]) / maxDim;
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    triangles: vertices.length / 3,
  };
}

function uploadMesh(mesh) {
  vertexCount = mesh.positions.length / 3;
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.position);
  gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.normal);
  gl.bufferData(gl.ARRAY_BUFFER, mesh.normals, gl.STATIC_DRAW);
}

function draw() {
  resizeCanvas();
  gl.enable(gl.DEPTH_TEST);
  gl.clearColor(0.02, 0.04, 0.09, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(program);

  const aspect = canvas.width / canvas.height;
  const projection = perspective(45 * Math.PI / 180, aspect, 0.01, 100);
  const model = multiply(
    translate(0, 0, -zoom),
    multiply(rotateX(rotationX), rotateY(rotationY)),
  );
  const mvp = multiply(projection, model);

  setMatrix("uModel", model);
  setMatrix("uMvp", mvp);
  gl.uniform3f(gl.getUniformLocation(program, "uLight"), 0.35, 0.55, 0.75);

  bindAttribute("aPosition", buffers.position);
  bindAttribute("aNormal", buffers.normal);
  gl.drawArrays(gl.TRIANGLES, 0, vertexCount);
  requestAnimationFrame(draw);
}

function createProgram() {
  const vertexSource = `
    attribute vec3 aPosition;
    attribute vec3 aNormal;
    uniform mat4 uMvp;
    uniform mat4 uModel;
    varying vec3 vNormal;
    varying vec3 vPosition;
    void main() {
      vNormal = mat3(uModel) * aNormal;
      vPosition = (uModel * vec4(aPosition, 1.0)).xyz;
      gl_Position = uMvp * vec4(aPosition, 1.0);
    }
  `;
  const fragmentSource = `
    precision mediump float;
    varying vec3 vNormal;
    varying vec3 vPosition;
    uniform vec3 uLight;
    void main() {
      vec3 normal = normalize(vNormal);
      float diffuse = max(dot(normal, normalize(uLight)), 0.0);
      float rim = pow(1.0 - max(dot(normal, normalize(-vPosition)), 0.0), 2.0);
      vec3 base = vec3(0.18, 0.64, 0.90);
      vec3 color = base * (0.25 + 0.75 * diffuse) + vec3(0.55, 0.85, 1.0) * rim * 0.3;
      gl_FragColor = vec4(color, 1.0);
    }
  `;
  const vertexShader = compileShader(gl.VERTEX_SHADER, vertexSource);
  const fragmentShader = compileShader(gl.FRAGMENT_SHADER, fragmentSource);
  const linked = gl.createProgram();
  gl.attachShader(linked, vertexShader);
  gl.attachShader(linked, fragmentShader);
  gl.linkProgram(linked);
  if (!gl.getProgramParameter(linked, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(linked));
  }
  return linked;
}

function compileShader(type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

function bindAttribute(name, buffer) {
  const location = gl.getAttribLocation(program, name);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 0, 0);
}

function setMatrix(name, matrix) {
  gl.uniformMatrix4fv(gl.getUniformLocation(program, name), false, new Float32Array(matrix));
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  const width = Math.floor(canvas.clientWidth * ratio);
  const height = Math.floor(canvas.clientHeight * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
    gl.viewport(0, 0, width, height);
  }
}

function installControls() {
  canvas.addEventListener("pointerdown", (event) => {
    dragging = true;
    lastMouse = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const dx = event.clientX - lastMouse[0];
    const dy = event.clientY - lastMouse[1];
    rotationY += dx * 0.01;
    rotationX += dy * 0.01;
    lastMouse = [event.clientX, event.clientY];
  });
  canvas.addEventListener("pointerup", () => {
    dragging = false;
  });
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoom = Math.max(0.9, Math.min(8, zoom + event.deltaY * 0.003));
  }, { passive: false });
  canvas.addEventListener("dblclick", () => {
    rotationX = -0.65;
    rotationY = 0.75;
    zoom = 2.65;
  });
}

function perspective(fovy, aspect, near, far) {
  const f = 1 / Math.tan(fovy / 2);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0,
  ];
}

function translate(x, y, z) {
  return [
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    x, y, z, 1,
  ];
}

function rotateX(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    1, 0, 0, 0,
    0, c, s, 0,
    0, -s, c, 0,
    0, 0, 0, 1,
  ];
}

function rotateY(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [
    c, 0, -s, 0,
    0, 1, 0, 0,
    s, 0, c, 0,
    0, 0, 0, 1,
  ];
}

function multiply(a, b) {
  const out = new Array(16).fill(0);
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      for (let k = 0; k < 4; k++) {
        out[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
      }
    }
  }
  return out;
}
