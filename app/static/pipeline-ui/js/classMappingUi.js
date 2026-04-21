/**
 * 类别映射可视化表单（参考 self_tools Converter 交互：行编辑 + 模式切换）。
 * 用于启动页与 discover_classes 审核点，产出 class_name_map / final_classes 或 class_index_map / training_names。
 */

function parseLines(text) {
  return String(text || "")
    .split(/[\r\n,，;；]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function uid() {
  return `cm_${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * @param {HTMLElement} host 挂载根（须空容器）
 * @param {{ radioName: string, seed?: object, discoveredNames?: string[] }} opts
 */
export function mountClassMappingForm(host, opts = {}) {
  const radioName = opts.radioName || `idxMode_${uid()}`;
  const seed = opts.seed || {};
  const discovered = opts.discoveredNames || [];

  host.innerHTML = "";
  host.classList.add("class-mapping-form");

  const intro = document.createElement("p");
  intro.className = "class-mapping-intro";
  intro.textContent =
    "为 XML 中的类名指定合并后的逻辑名，再选择索引方式（与 xml-to-yolo 一致）。";
  host.appendChild(intro);

  const rowWrap = document.createElement("div");
  rowWrap.dataset.role = "name-rows";
  rowWrap.className = "class-map-rows";
  const rowHeader = document.createElement("div");
  rowHeader.className = "class-map-row class-map-row--header";
  rowHeader.innerHTML = "<span>XML 原名</span><span></span><span>合并为逻辑名</span><span></span>";
  rowWrap.appendChild(rowHeader);

  const addNameRow = (fromVal = "", toVal = "") => {
    const row = document.createElement("div");
    row.className = "class-map-row";
    const from = document.createElement("input");
    from.type = "text";
    from.placeholder = "如 louyou1";
    from.value = fromVal;
    from.dataset.part = "from";
    const arrow = document.createElement("span");
    arrow.className = "class-map-arrow";
    arrow.textContent = "→";
    const to = document.createElement("input");
    to.type = "text";
    to.placeholder = "如 louyou";
    to.value = toVal;
    to.dataset.part = "to";
    const del = document.createElement("button");
    del.type = "button";
    del.className = "ghost mini class-map-row-del";
    del.textContent = "删";
    del.addEventListener("click", () => {
      row.remove();
    });
    row.append(from, arrow, to, del);
    rowWrap.appendChild(row);
  };

  const mapSeed = seed.class_name_map && typeof seed.class_name_map === "object" ? seed.class_name_map : null;
  if (mapSeed && Object.keys(mapSeed).length) {
    for (const [k, v] of Object.entries(mapSeed)) {
      addNameRow(k, v);
    }
  } else if (discovered.length) {
    for (const n of discovered) {
      addNameRow(n, n);
    }
  } else {
    addNameRow("", "");
  }

  const addBtn = document.createElement("button");
  addBtn.type = "button";
  addBtn.className = "ghost mini class-map-add";
  addBtn.textContent = "+ 添加重命名行";
  addBtn.addEventListener("click", () => addNameRow("", ""));

  host.appendChild(rowWrap);
  host.appendChild(addBtn);

  const modeBox = document.createElement("div");
  modeBox.className = "class-index-mode";
  const modeLabel = document.createElement("div");
  modeLabel.className = "class-index-mode-label";
  modeLabel.textContent = "索引与训练名";
  const pills = document.createElement("div");
  pills.className = "mode-pills";

  const rOrdered = document.createElement("input");
  rOrdered.type = "radio";
  rOrdered.name = radioName;
  rOrdered.value = "ordered";
  rOrdered.id = `${radioName}_ord`;
  const rExplicit = document.createElement("input");
  rExplicit.type = "radio";
  rExplicit.name = radioName;
  rExplicit.value = "explicit";
  rExplicit.id = `${radioName}_exp`;

  const hasExplicit =
    seed.class_index_map &&
    typeof seed.class_index_map === "object" &&
    Object.keys(seed.class_index_map).length > 0;

  if (hasExplicit) {
    rExplicit.checked = true;
  } else {
    rOrdered.checked = true;
  }

  const l1 = document.createElement("label");
  l1.htmlFor = rOrdered.id;
  l1.append(rOrdered, document.createTextNode(" 按目标类列表顺序（final_classes）"));
  const l2 = document.createElement("label");
  l2.htmlFor = rExplicit.id;
  l2.append(rExplicit, document.createTextNode(" 显式类别 id（class_index_map）"));
  pills.append(l1, l2);

  const panelOrd = document.createElement("div");
  panelOrd.className = "mode-panel";
  panelOrd.dataset.role = "panel-ordered";
  const taOrd = document.createElement("textarea");
  taOrd.rows = 3;
  taOrd.placeholder = "每行一个类名，自上而下为 id 0,1,2…";
  taOrd.dataset.role = "final-classes";
  if (Array.isArray(seed.final_classes) && seed.final_classes.length) {
    taOrd.value = seed.final_classes.join("\n");
  }
  const lbOrd = document.createElement("label");
  lbOrd.textContent = "目标类顺序";
  panelOrd.append(lbOrd, taOrd);

  const panelExp = document.createElement("div");
  panelExp.className = "mode-panel";
  panelExp.dataset.role = "panel-explicit";
  const idxRows = document.createElement("div");
  idxRows.dataset.role = "index-rows";
  idxRows.className = "class-map-rows";
  const idxHead = document.createElement("div");
  idxHead.className = "class-map-row class-map-row--header";
  idxHead.innerHTML = "<span>逻辑类名</span><span></span><span>YOLO id</span><span></span>";
  idxRows.appendChild(idxHead);

  const addIdxRow = (nameVal = "", idVal = "") => {
    const row = document.createElement("div");
    row.className = "class-map-row";
    const nm = document.createElement("input");
    nm.type = "text";
    nm.placeholder = "逻辑名";
    nm.value = nameVal;
    nm.dataset.part = "idx-name";
    const arrow = document.createElement("span");
    arrow.className = "class-map-arrow";
    arrow.textContent = "→";
    const idIn = document.createElement("input");
    idIn.type = "number";
    idIn.min = "0";
    idIn.step = "1";
    idIn.placeholder = "0";
    idIn.value = idVal === "" || idVal === undefined ? "" : String(idVal);
    idIn.dataset.part = "idx-id";
    const del = document.createElement("button");
    del.type = "button";
    del.className = "ghost mini";
    del.textContent = "删";
    del.addEventListener("click", () => row.remove());
    row.append(nm, arrow, idIn, del);
    idxRows.appendChild(row);
  };

  const idxMapSeed =
    seed.class_index_map && typeof seed.class_index_map === "object" ? seed.class_index_map : null;
  if (idxMapSeed && Object.keys(idxMapSeed).length) {
    for (const [k, v] of Object.entries(idxMapSeed)) {
      addIdxRow(k, v);
    }
  } else {
    addIdxRow("", "");
  }

  const addIdxBtn = document.createElement("button");
  addIdxBtn.type = "button";
  addIdxBtn.className = "ghost mini";
  addIdxBtn.textContent = "+ 添加 逻辑名→id";
  addIdxBtn.addEventListener("click", () => addIdxRow("", ""));

  const taTn = document.createElement("textarea");
  taTn.rows = 2;
  taTn.placeholder = "可选：每行对应 id 0,1,…（training_names），留空则自动生成";
  taTn.dataset.role = "training-names";
  if (Array.isArray(seed.training_names) && seed.training_names.length) {
    taTn.value = seed.training_names.join("\n");
  }
  const lbTn = document.createElement("label");
  lbTn.textContent = "训练显示名（可选）";
  panelExp.append(idxRows, addIdxBtn, lbTn, taTn);

  const syncPanels = () => {
    const exp = rExplicit.checked;
    panelOrd.hidden = exp;
    panelExp.hidden = !exp;
  };
  rOrdered.addEventListener("change", syncPanels);
  rExplicit.addEventListener("change", syncPanels);
  syncPanels();

  modeBox.append(modeLabel, pills, panelOrd, panelExp);
  host.appendChild(modeBox);

  host._radioOrdered = rOrdered;
  host._radioExplicit = rExplicit;
}

/**
 * @param {HTMLElement} host mountClassMappingForm 挂载的根
 * @returns {Record<string, unknown>}
 */
export function collectClassMappingPayload(host) {
  const out = {};

  const nameRows = host.querySelectorAll('[data-role="name-rows"] .class-map-row:not(.class-map-row--header)');
  const nameMap = {};
  nameRows.forEach((row) => {
    const from = row.querySelector('[data-part="from"]')?.value?.trim() ?? "";
    const to = row.querySelector('[data-part="to"]')?.value?.trim() ?? "";
    if (from) {
      nameMap[from] = to || from;
    }
  });
  if (Object.keys(nameMap).length) {
    out.class_name_map = nameMap;
  }

  const explicit = host._radioExplicit?.checked;
  if (!explicit) {
    const lines = parseLines(host.querySelector('[data-role="final-classes"]')?.value || "");
    if (lines.length) {
      out.final_classes = lines;
    }
  } else {
    const idxRows = host.querySelectorAll(
      '[data-role="index-rows"] .class-map-row:not(.class-map-row--header)',
    );
    const idxMap = {};
    idxRows.forEach((row) => {
      const name = row.querySelector('[data-part="idx-name"]')?.value?.trim() ?? "";
      const rawId = row.querySelector('[data-part="idx-id"]')?.value?.trim() ?? "";
      if (!name || rawId === "") return;
      const id = Number.parseInt(rawId, 10);
      if (Number.isFinite(id) && id >= 0) {
        idxMap[name] = id;
      }
    });
    if (Object.keys(idxMap).length) {
      out.class_index_map = idxMap;
    }
    const tnLines = parseLines(host.querySelector('[data-role="training-names"]')?.value || "");
    if (tnLines.length) {
      out.training_names = tnLines;
    }
  }

  return out;
}
