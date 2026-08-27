/** Dynamic scene UI: hide unused scene_* widgets; + Add scene / Update scenes. */
const { app } = window.comfyAPI.app;

const NODE_NAMES = new Set([
  "KreaCharSheetScenes",
  "KreaMultiShotStills",
  "LTX25MultiSceneMSR",
]);
const MAX_SCENES = 8;

function sceneIndexFromWidgetName(name) {
  const m = /^scene_(\d+)_(first|last|motion|prompt)$/.exec(name || "");
  return m ? parseInt(m[1], 10) : null;
}

function applySceneVisibility(node) {
  if (!node.widgets) return;
  const countW = node.widgets.find((w) => w.name === "scene_count");
  if (!countW) return;
  const count = Math.max(1, Math.min(MAX_SCENES, countW.value | 0));
  for (const w of node.widgets) {
    const idx = sceneIndexFromWidgetName(w.name);
    if (idx == null) continue;
    const hide = idx > count;
    if (hide) {
      w.computeSize = () => [0, -4];
      w._kreaHidden = true;
    } else if (w._kreaHidden) {
      delete w.computeSize;
      delete w._kreaHidden;
    }
  }
  try {
    node.setSize([node.size[0], node.computeSize()[1]]);
  } catch (_) {
    /* ignore */
  }
  node.setDirtyCanvas?.(true, true);
}

function ensureButtons(node) {
  if (node._kreaSceneButtons) return;
  node._kreaSceneButtons = true;

  node.addWidget("button", "+ Add scene", null, () => {
    const countW = node.widgets?.find((w) => w.name === "scene_count");
    if (!countW) return;
    if (countW.value >= MAX_SCENES) return;
    countW.value = countW.value + 1;
    if (countW.callback) countW.callback(countW.value, app.canvas);
    applySceneVisibility(node);
  });

  node.addWidget("button", "Update scenes", null, () => {
    applySceneVisibility(node);
  });

  const countW = node.widgets?.find((w) => w.name === "scene_count");
  if (countW) {
    const orig = countW.callback;
    countW.callback = function (value, canvas) {
      const r = orig ? orig.apply(this, arguments) : undefined;
      applySceneVisibility(node);
      return r;
    };
  }
}

app.registerExtension({
  name: "KreaCharSheetScenes.dynamicScenes",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData?.name)) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      ensureButtons(this);
      setTimeout(() => applySceneVisibility(this), 0);
      return r;
    };
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => {
        ensureButtons(this);
        applySceneVisibility(this);
      }, 0);
      return r;
    };
  },
});
