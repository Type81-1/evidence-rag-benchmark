const $ = selector => document.querySelector(selector);
const labels = {
  baseline: ['纯通用模型', '不检索，依赖模型既有知识'],
  good: ['优质 RAG', '召回主题匹配的指南与研究'],
  noisy: ['噪声 RAG', '混入高可信度但不相关的材料'],
  missing: ['缺失 RAG', '知识库未召回可用证据']
};
const metricLabels = {claim_coverage:'主张覆盖', citation_precision:'引用精确', citation_coverage:'引用覆盖', unsupported_citation_rate:'无支撑引用', appropriate_refusal:'合理拒答'};
let currentDomain = 'nutrition';
const domainCopy = {
  nutrition: {eyebrow:'TRACK 03 · NUTRITION RAG BENCHMARK', title:'营养指南 RAG，真的更好吗？', subtitle:'把“更专业”拆成可测量的引用、完整性与失败边界。', rag:'营养指南 RAG'},
  hypertension: {eyebrow:'TRACK 03 · HYPERTENSION RAG BENCHMARK', title:'高血压文献 RAG，优势在哪里？', subtitle:'比较通用知识与可追溯临床证据，并主动测试错误检索和拒答边界。', rag:'高血压文献 RAG'}
};

function escapeHtml(value='') { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c])); }
function renderMarkdown(value) {
  return escapeHtml(value)
    .replace(/^## (.+)$/gm, '<h4>$1</h4>')
    .replace(/^- (.+)$/gm, '<p class="bullet">$1</p>')
    .replace(/\[([A-Z]\d+)]/g, '<mark>[$1]</mark>')
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/^/, '<p>').replace(/$/, '</p>');
}
function percent(value) { return `${Math.round(value * 100)}%`; }

function metricRows(metrics) {
  return Object.entries(metrics).map(([key, value]) => `<div class="mini-metric ${key === 'unsupported_citation_rate' && value > 0 ? 'risk' : ''}"><span>${metricLabels[key]}</span><b>${percent(value)}</b></div>`).join('');
}

async function loadOverview() {
  const response = await fetch(`/api/benchmark?domain=${currentDomain}`);
  const data = await response.json();
  const order = ['baseline','good','noisy','missing'];
  $('#summary-grid').innerHTML = order.map(key => {
    const item = data.summary[key];
    return `<article class="summary-card ${key}"><header><i></i><div><h3>${labels[key][0]}</h3><p>${labels[key][1]}</p></div></header><div class="score"><strong>${percent(item.claim_coverage)}</strong><span>主张覆盖率</span></div><dl><div><dt>引用精确</dt><dd>${percent(item.citation_precision)}</dd></div><div><dt>引用覆盖</dt><dd>${percent(item.citation_coverage)}</dd></div><div><dt>合理拒答</dt><dd>${percent(item.appropriate_refusal)}</dd></div></dl></article>`;
  }).join('');
  const good = data.summary.good, noisy = data.summary.noisy;
  $('#finding-strip').innerHTML = `<b>离线管线观察</b><span>优质 RAG 的引用精确率为 ${percent(good.citation_precision)}；噪声检索后为 ${percent(noisy.citation_precision)}，主张覆盖率为 ${percent(noisy.claim_coverage)}。</span><em>这验证了实验能捕捉“检索拖累”，不是最终模型结论。</em>`;
}

async function loadQuestions() {
  const response = await fetch(`/api/questions?domain=${currentDomain}`);
  const questions = await response.json();
  $('#question-select').innerHTML = questions.map(item => `<option value="${item.id}">${item.id} · ${escapeHtml(item.question)}</option>`).join('');
}

async function runComparison() {
  const button = $('#run');
  const notice = $('#notice');
  button.disabled = true;
  notice.hidden = false;
  notice.textContent = '正在运行两条回答路径并计算指标…';
  try {
    const response = await fetch('/api/compare', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({domain:currentDomain, question_id:$('#question-select').value, retrieval_condition:$('#condition-select').value, live:$('#live-mode').checked})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || '运行失败');
    $('#mode-label').textContent = data.run_mode === 'live_model' ? '真实模型运行' : '可复现实验样例';
    notice.textContent = `${data.question.id} · ${data.question.topic} · ${labels[data.condition][0]} · 对照完成`;
    $('#baseline-metrics').innerHTML = metricRows(data.baseline.metrics);
    $('#rag-metrics').innerHTML = metricRows(data.rag.metrics);
    $('#baseline-answer').innerHTML = renderMarkdown(data.baseline.answer);
    $('#rag-answer').innerHTML = renderMarkdown(data.rag.answer);
    $('#rag-source-pill').textContent = `${data.rag.evidence.length} 条检索证据`;
    $('#evidence-list').innerHTML = data.rag.evidence.length ? `<h4>检索上下文</h4>${data.rag.evidence.map(item => `<a href="${item.url}" target="_blank" rel="noreferrer"><b>[${item.id}] ${escapeHtml(item.title)}</b><span>${escapeHtml(item.organization)} · ${item.year} · ${escapeHtml(item.quality)}</span></a>`).join('')}` : '<div class="empty-evidence">没有召回证据，系统应当拒答。</div>';
    $('#comparison').hidden = false;
  } catch (error) { notice.textContent = `无法运行：${error.message}`; }
  finally { button.disabled = false; }
}

async function switchDomain(domain) {
  if (domain === currentDomain && $('#question-select').options.length) return;
  currentDomain = domain;
  const copy = domainCopy[domain];
  document.querySelectorAll('[data-domain]').forEach(button => button.classList.toggle('active', button.dataset.domain === domain));
  $('#domain-eyebrow').textContent = copy.eyebrow;
  $('#domain-title').textContent = copy.title;
  $('#domain-subtitle').textContent = copy.subtitle;
  $('#rag-panel-title').textContent = copy.rag;
  $('#summary-grid').innerHTML = '';
  $('#comparison').hidden = true;
  $('#finding-strip').textContent = '正在计算实验总览…';
  await Promise.all([loadOverview(), loadQuestions()]);
  await runComparison();
}

$('#run').addEventListener('click', runComparison);
document.querySelectorAll('[data-domain]').forEach(button => button.addEventListener('click', () => switchDomain(button.dataset.domain)));
Promise.all([loadOverview(), loadQuestions()]).then(runComparison).catch(error => { $('#finding-strip').textContent = `加载失败：${error.message}`; });
