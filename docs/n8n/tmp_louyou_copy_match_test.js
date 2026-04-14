const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const XLSX_PATH = '/media/qzq/16T/20260413TEDS非转数据登记表.xlsx';
const SHEET_NAME = '数据表';
const WEBHOOK_URL = 'http://192.168.2.26:5678/webhook/tool-diban-louyou-empty-rows-by-path';
const SOURCE_ROOT = '/media/qzq/16T/download-2026-04-13_16-14-47/teds/整车图/正线';
const DEST_ROOT = '/media/qzq/16T/download-2026-04-13_16-14-47/teds/整车图_整理';
const MAX_PREVIEW = 20;
const INCLUDE_UNREGISTERED_FOLDERS = true;
const BUREAU_ALIAS = {
  哈尔滨局: ['哈局'],
  哈局: ['哈尔滨局'],
};

function normalizeText(s) {
  return String(s || '').trim();
}

function safeDirEntries(dir) {
  try {
    return fs.readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name);
  } catch {
    return [];
  }
}

function toDateNum(yyyymmdd) {
  const s = String(yyyymmdd || '').replace(/[^\d]/g, '');
  if (!/^\d{8}$/.test(s)) return null;
  return Number(s);
}

function buildDate(y, m, d) {
  const yy = String(y).padStart(4, '0');
  const mm = String(m).padStart(2, '0');
  const dd = String(d).padStart(2, '0');
  return Number(`${yy}${mm}${dd}`);
}

function parseRangeFromName(name) {
  const text = normalizeText(name);
  if (!text) return null;

  // 20260403-20260408
  const fullRange = text.match(/(20\d{6})\s*[-_~至到]+\s*(20\d{6})/);
  if (fullRange) {
    const start = toDateNum(fullRange[1]);
    const end = toDateNum(fullRange[2]);
    if (start && end) return [Math.min(start, end), Math.max(start, end)];
  }

  // 20260403-0406
  const shortRange = text.match(/(20\d{2})(\d{2})(\d{2})\s*[-_~至到]+\s*(\d{2})(\d{2})/);
  if (shortRange) {
    const y = shortRange[1];
    const m1 = shortRange[2];
    const d1 = shortRange[3];
    const m2 = shortRange[4];
    const d2 = shortRange[5];
    const start = buildDate(y, m1, d1);
    const end = buildDate(y, m2, d2);
    return [Math.min(start, end), Math.max(start, end)];
  }

  // 单日期 20260407
  const one = text.match(/(20\d{6})/);
  if (one) {
    const v = toDateNum(one[1]);
    if (v) return [v, v];
  }
  return null;
}

function rangeOverlap(a, b) {
  if (!a || !b) return false;
  return a[0] <= b[1] && b[0] <= a[1];
}

function fetchMarkedRangesFromExcel(xlsxPath, sheetName) {
  const pyCode = `
import json
import pandas as pd

xlsx_path = ${JSON.stringify(XLSX_PATH)}
sheet_name = ${JSON.stringify(SHEET_NAME)}
df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
need = ["数据来源", "文件夹名称", "底板漏油"]
for k in need:
    if k not in df.columns:
        raise Exception(f"missing column: {k}")

out = []
for _, row in df.iterrows():
    bureau = row["数据来源"]
    folder = row["文件夹名称"]
    val = row["底板漏油"]
    if pd.isna(val):
        continue
    s = str(val).strip()
    if not s:
        continue
    out.append({
        "sourceName": "" if pd.isna(bureau) else str(bureau).strip(),
        "folderName": "" if pd.isna(folder) else str(folder).strip(),
        "markValue": s
    })
print(json.dumps(out, ensure_ascii=False))
`;

  try {
    const raw = execFileSync('python', ['-c', pyCode], { encoding: 'utf8' });
    const data = JSON.parse(raw);
    return Array.isArray(data) ? data : [];
  } catch (e) {
    console.error('读取已标记区间失败，将不启用避让规则:', e.message);
    return [];
  }
}

function bureauCandidates(sourceName, rootDirs) {
  const src = normalizeText(sourceName).replace(/\s+/g, '');
  const srcNoJu = src.replace(/局/g, '');
  const aliasList = [src, ...(BUREAU_ALIAS[src] || [])]
    .map((x) => normalizeText(x).replace(/\s+/g, ''))
    .filter(Boolean);
  return rootDirs.filter((d) => {
    const dn = d.replace(/\s+/g, '');
    const dnNoJu = dn.replace(/局/g, '');
    return aliasList.some((a) => {
      const aNoJu = a.replace(/局/g, '');
      return dn === a || dn.includes(a) || a.includes(dn) || dnNoJu === aNoJu || dnNoJu.includes(aNoJu);
    });
  });
}

function pickFolderMatch(targetFolderName, children) {
  const t = normalizeText(targetFolderName);
  if (!t) return null;

  // 1) 精确匹配
  const exact = children.find((c) => normalizeText(c) === t);
  if (exact) return { matchType: 'exact', folder: exact };

  // 2) 包含匹配
  const contains = children.find((c) => normalizeText(c).includes(t) || t.includes(normalizeText(c)));
  if (contains) return { matchType: 'contains', folder: contains };

  // 3) 日期区间重叠匹配
  const tr = parseRangeFromName(t);
  if (tr) {
    const dateMatched = children
      .map((c) => ({ c, r: parseRangeFromName(c) }))
      .filter((x) => x.r && rangeOverlap(tr, x.r))
      .sort((a, b) => a.r[0] - b.r[0]);
    if (dateMatched.length > 0) {
      return { matchType: 'date-range-overlap', folder: dateMatched[0].c };
    }
  }
  return null;
}

async function fetchRows() {
  const resp = await fetch(WEBHOOK_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ xlsxPath: XLSX_PATH, sheetName: SHEET_NAME }),
  });
  if (!resp.ok) throw new Error(`Webhook error: ${resp.status}`);
  const data = await resp.json();
  const item = Array.isArray(data) ? data[0] : data;
  return item || {};
}

function ensureDestDir(destPath) {
  fs.mkdirSync(path.dirname(destPath), { recursive: true });
}

function copyFolder(src, dest) {
  ensureDestDir(dest);
  fs.cpSync(src, dest, { recursive: true, force: false, errorOnExist: false });
}

async function main() {
  const doCopy = process.env.DO_COPY === '1';
  const payload = await fetchRows();
  const rows = Array.isArray(payload.emptyRows) ? payload.emptyRows : [];
  const markedRows = fetchMarkedRangesFromExcel(XLSX_PATH, SHEET_NAME);

  const bureauDirs = safeDirEntries(SOURCE_ROOT);
  const results = [];
  const failedTableItems = [];
  const copiedOrPlannedSourceFolders = new Set();

  const markedRangesByBureau = {};
  for (const r of markedRows) {
    const b = normalizeText(r.sourceName);
    const f = normalizeText(r.folderName);
    if (!b || !f) continue;
    const rg = parseRangeFromName(f);
    if (!rg) continue;
    if (!markedRangesByBureau[b]) markedRangesByBureau[b] = [];
    markedRangesByBureau[b].push({ range: rg, folderName: f, markValue: r.markValue || '' });
  }

  for (const row of rows) {
    const sourceName = normalizeText(row['数据来源']);
    const folderName = normalizeText(row['文件夹名称']);
    const filePath = normalizeText(row['文件路径']);

    // 有明确文件路径的先跳过（你可以后续合并逻辑）
    if (filePath) {
      const item = {
        ok: false,
        reason: '已有文件路径，当前测试脚本只处理文件路径为空的反查场景',
        sourceName,
        folderName,
      };
      results.push(item);
      failedTableItems.push({
        登记日期: row['登记日期'] || '',
        数据来源: sourceName,
        文件夹名称: folderName,
        文件路径: filePath || '',
        失败原因: item.reason,
      });
      continue;
    }

    const bureauMatches = bureauCandidates(sourceName, bureauDirs);
    if (bureauMatches.length === 0) {
      const item = { ok: false, reason: '未找到对应路局目录', sourceName, folderName };
      results.push(item);
      failedTableItems.push({
        登记日期: row['登记日期'] || '',
        数据来源: sourceName,
        文件夹名称: folderName,
        文件路径: filePath || '',
        失败原因: item.reason,
      });
      continue;
    }

    let matched = null;
    for (const bureau of bureauMatches) {
      const bureauPath = path.join(SOURCE_ROOT, bureau);
      const children = safeDirEntries(bureauPath);
      const picked = pickFolderMatch(folderName, children);
      if (picked) {
        matched = { bureau, bureauPath, picked };
        break;
      }
    }

    if (!matched) {
      const item = { ok: false, reason: '路局目录下未匹配到子目录', sourceName, folderName, bureauMatches };
      results.push(item);
      failedTableItems.push({
        登记日期: row['登记日期'] || '',
        数据来源: sourceName,
        文件夹名称: folderName,
        文件路径: filePath || '',
        失败原因: item.reason,
      });
      continue;
    }

    const src = path.join(matched.bureauPath, matched.picked.folder);
    const rel = path.relative(SOURCE_ROOT, src);
    const dest = path.join(DEST_ROOT, rel);

    if (doCopy) {
      try {
        copyFolder(src, dest);
        results.push({ ok: true, executed: true, matchType: matched.picked.matchType, sourceName, folderName, sourceFolder: src, destinationFolder: dest });
        copiedOrPlannedSourceFolders.add(src);
      } catch (e) {
        const item = { ok: false, executed: true, reason: e.message, sourceName, folderName, sourceFolder: src, destinationFolder: dest };
        results.push(item);
        failedTableItems.push({
          登记日期: row['登记日期'] || '',
          数据来源: sourceName,
          文件夹名称: folderName,
          文件路径: filePath || '',
          失败原因: item.reason,
        });
      }
    } else {
      results.push({ ok: true, executed: false, matchType: matched.picked.matchType, sourceName, folderName, sourceFolder: src, destinationFolder: dest });
      copiedOrPlannedSourceFolders.add(src);
    }
  }

  // 补充“表格未登记但目录中存在”的数据
  if (INCLUDE_UNREGISTERED_FOLDERS) {
    for (const bureau of bureauDirs) {
      const bureauPath = path.join(SOURCE_ROOT, bureau);
      const children = safeDirEntries(bureauPath);
      const normalizedBureau = normalizeText(bureau);
      const markedCandidates = [
        ...(markedRangesByBureau[normalizedBureau] || []),
        ...(Object.entries(BUREAU_ALIAS)
          .filter(([k]) => normalizeText(k) === normalizedBureau)
          .flatMap(([_, aliases]) => aliases)
          .flatMap((alias) => markedRangesByBureau[normalizeText(alias)] || [])),
      ];
      for (const child of children) {
        const childName = normalizeText(child);
        if (!/(已标注|漏报)/.test(childName)) continue;
        const src = path.join(bureauPath, childName);
        if (copiedOrPlannedSourceFolders.has(src)) continue;

        const childRange = parseRangeFromName(childName);
        if (childRange && markedCandidates.length > 0) {
          const covered = markedCandidates.find((m) => rangeOverlap(childRange, m.range));
          if (covered) {
            results.push({
              ok: false,
              executed: false,
              isSupplement: true,
              reason: `补充跳过：区间与已标记项目重叠(${covered.folderName} => ${covered.markValue})`,
              sourceName: bureau,
              folderName: childName,
              sourceFolder: src,
            });
            continue;
          }
        }

        const rel = path.relative(SOURCE_ROOT, src);
        const dest = path.join(DEST_ROOT, rel);
        if (doCopy) {
          try {
            copyFolder(src, dest);
            results.push({
              ok: true,
              executed: true,
              isSupplement: true,
              reason: '目录存在但表格未登记，自动补充',
              sourceName: bureau,
              folderName: childName,
              sourceFolder: src,
              destinationFolder: dest,
            });
            copiedOrPlannedSourceFolders.add(src);
          } catch (e) {
            results.push({
              ok: false,
              executed: true,
              isSupplement: true,
              reason: `补充拷贝失败: ${e.message}`,
              sourceName: bureau,
              folderName: childName,
              sourceFolder: src,
              destinationFolder: dest,
            });
          }
        } else {
          results.push({
            ok: true,
            executed: false,
            isSupplement: true,
            reason: '目录存在但表格未登记，自动补充',
            sourceName: bureau,
            folderName: childName,
            sourceFolder: src,
            destinationFolder: dest,
          });
          copiedOrPlannedSourceFolders.add(src);
        }
      }
    }
  }

  const ok = results.filter((r) => r.ok).length;
  const failed = results.length - ok;
  const summary = {
    doCopy,
    includeUnregisteredFolders: INCLUDE_UNREGISTERED_FOLDERS,
    xlsxPath: XLSX_PATH,
    sheetName: SHEET_NAME,
    totalRows: payload.totalRows,
    emptyCount: payload.emptyCount,
    handledRows: rows.length,
    matched: ok,
    failed,
    supplements: results.filter((r) => r.isSupplement && r.ok).length,
  };

  console.log('=== SUMMARY ===');
  console.log(JSON.stringify(summary, null, 2));
  console.log('=== PREVIEW ===');
  console.log(JSON.stringify(results.slice(0, MAX_PREVIEW), null, 2));
  if (results.length > MAX_PREVIEW) {
    console.log(`... 其余 ${results.length - MAX_PREVIEW} 条已省略`);
  }
  console.log('=== FAILED_TABLE_ITEMS ===');
  console.log(JSON.stringify(failedTableItems, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
