const $ = selector => document.querySelector(selector);
const rubricKeys = ['correctness','completeness','safety','clarity','citation_quality','refusal_quality'];
const metricLabels = {correctness:'正确性',completeness:'完整性',safety:'安全性',clarity:'清晰度',citation_quality:'引用质量',refusal_quality:'拒答质量'};
const armLabels = {baseline:['A · 裸模型','无检索，仅模型知识'],good:['B · 正常 RAG','BM25 + TF-IDF 融合召回'],noisy:['C · 劣化 RAG','注入低相关证据'],missing:['D · 检索缺失','空证据包与拒答测试']};
const domainCopy = {
  nutrition:{eyebrow:'TRACK 03 · NUTRITION RAG BENCHMARK',title:'营养指南 RAG，真的更好吗？',subtitle:'预注册题集、锁定变量，再检验引用、质量和失败边界。',rag:'营养指南 RAG'},
  hypertension:{eyebrow:'TRACK 03 · HYPERTENSION RAG BENCHMARK',title:'高血压文献 RAG，优势在哪里？',subtitle:'同模型同 Prompt，对照可追溯证据、噪声召回与安全拒答。',rag:'高血压文献 RAG'}
};
let currentDomain = 'nutrition';
let currentComparison = null;
let currentView = 'design';
let displayedOutputs = {X:'baseline',Y:'rag'};

function escapeHtml(value=''){return String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));}
function percent(value){return `${Math.round(Number(value||0)*100)}%`;}
async function responseJson(response){const text=await response.text();try{return text?JSON.parse(text):{};}catch{return {detail:{code:`HTTP_${response.status}`,message:text.slice(0,240)||'服务返回了非 JSON 错误'}};}}
function errorMessage(data,fallback='运行失败'){const detail=data?.detail;return typeof detail==='string'?detail:(detail?.message||fallback);}
function renderMarkdown(value){return escapeHtml(value).replace(/^## (.+)$/gm,'<h4>$1</h4>').replace(/^- (.+)$/gm,'<p class="bullet">$1</p>').replace(/\[([A-Z][A-Z0-9_-]*)]/g,'<mark>[$1]</mark>').replace(/\n{2,}/g,'</p><p>').replace(/^/,'<p>').replace(/$/,'</p>');}
function metricRows(metrics){return rubricKeys.map(key=>`<div class="mini-metric"><span>${metricLabels[key]}</span><b>${percent(metrics[key])}</b></div>`).join('');}

async function loadStatus(){
  const data = await fetch('/api/project-status').then(r=>r.json());
  $('#run-meta').textContent = `${data.model} · reasoning ${data.reasoning_effort} · ${data.prompt_version} · ${data.rubric_version}`;
  $('#live-help').textContent = data.live_model_available ? `${data.model} 已配置` : '需要 API Key';
  $('#model-select').value=data.model;$('#reasoning-effort').value=data.reasoning_effort;
  if(data.proxy?.configured&&data.proxy.reachable===false)$('#model-config-status').textContent='Windows 代理已配置，但代理客户端未监听；请先启动代理客户端';
}

async function loadOverview(){
  const data = await fetch(`/api/benchmark?domain=${currentDomain}`).then(r=>r.json());
  $('#question-count').textContent = data.question_count;
  $('#summary-grid').innerHTML = ['baseline','good','noisy','missing'].map(key=>{
    const item=data.summary[key];
    return `<article class="summary-card ${key}"><header><i></i><div><h3>${armLabels[key][0]}</h3><p>${armLabels[key][1]}</p></div></header><div class="score"><strong>${percent(item.correctness)}</strong><span>机械正确性代理</span></div><dl>${rubricKeys.slice(1).map(metric=>`<div><dt>${metricLabels[metric]}</dt><dd>${percent(item[metric])}</dd></div>`).join('')}</dl></article>`;
  }).join('');
  const good=data.summary.good,noisy=data.summary.noisy;
  $('#finding-strip').innerHTML=`<b>管线自检</b><span>正常 RAG 引用质量代理 ${percent(good.citation_quality)}；劣化 RAG ${percent(noisy.citation_quality)}。</span><em>这些是机械代理值，正式结论以真实模型重复实验和盲评为准。</em>`;
}

async function loadQuestions(){
  const selected=$('#question-select').value;
  const endpoint=currentView==='design'?'/api/design/questions':'/api/questions';
  const questions=await fetch(`${endpoint}?domain=${currentDomain}`).then(r=>r.json());
  $('#question-select').innerHTML=questions.map(item=>{
    let tag='';
    if(currentView==='design'&&item.urgent)tag=' · [急症升级]';
    else if(currentView==='design'&&item.should_abstain)tag=` · [${String(item.notes).includes('证据不足')?'证据不足':'越界拒答'}]`;
    return `<option value="${item.id}">${item.id} · ${escapeHtml(item.question)}${tag}</option>`;
  }).join('');
  if(selected&&[...$('#question-select').options].some(option=>option.value===selected))$('#question-select').value=selected;
}

function renderOutputs(data){
  const swap=currentView==='blind'&&parseInt(data.comparison_id.slice(-1),16)%2===1;
  displayedOutputs=swap?{X:'rag',Y:'baseline'}:{X:'baseline',Y:'rag'};
  const left=data[displayedOutputs.X],right=data[displayedOutputs.Y];
  $('#baseline-metrics').innerHTML=metricRows(left.metrics);$('#rag-metrics').innerHTML=metricRows(right.metrics);
  $('#baseline-answer').innerHTML=renderMarkdown(left.answer);$('#rag-answer').innerHTML=renderMarkdown(right.answer);
  if(currentView==='blind'){
    $('#output-x-tag').textContent='ANONYMOUS OUTPUT X';$('#output-x-title').textContent='匿名输出 X';$('#output-x-pill').textContent='身份已隐藏';
    $('#output-y-tag').textContent='ANONYMOUS OUTPUT Y';$('#rag-panel-title').textContent='匿名输出 Y';$('#rag-source-pill').textContent='身份已隐藏';
  }else{
    displayedOutputs={X:'baseline',Y:'rag'};
    $('#output-x-tag').textContent='ARM A';$('#output-x-title').textContent='纯通用模型';$('#output-x-pill').textContent='证据包为空';
    $('#output-y-tag').textContent='ARM B/C/D';$('#rag-panel-title').textContent=domainCopy[currentDomain].rag;$('#rag-source-pill').textContent=`${data.rag.evidence.length} 条证据`;
  }
}

async function setView(view){
  currentView=view;document.body.classList.toggle('blind-mode',view==='blind');
  document.querySelectorAll('[data-view]').forEach(button=>button.classList.toggle('active',button.dataset.view===view));
  $('#view-mode-note').textContent=view==='blind'?'盲评模式隐藏场景标签、系统身份、检索诊断和机械分数，并随机交换 X/Y。':'实验操作模式显示场景标签、系统身份和检索诊断。';
  await loadQuestions();if(currentComparison)renderOutputs(currentComparison);
}

async function runComparison(){
  const button=$('#run'),notice=$('#notice');button.disabled=true;notice.hidden=false;notice.textContent='正在运行锁定 Prompt 的两条路径…';
  try{
    const response=await fetch('/api/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:currentDomain,question_id:$('#question-select').value,retrieval_condition:$('#condition-select').value,live:$('#live-mode').checked})});
    const data=await responseJson(response);if(!response.ok)throw new Error(errorMessage(data));
    currentComparison=data;
    $('#mode-label').textContent=data.run_mode==='live_model'?'真实模型运行':'管线演示 · 非模型结论';
    notice.textContent=`${data.question.id} · ${data.question.topic} · ${armLabels[data.condition][0]} · Prompt ${data.prompt_version}`;
    renderOutputs(data);
    const rm=data.rag.retrieval_metrics;$('#retrieval-diagnostic').hidden=false;$('#retrieval-diagnostic').innerHTML=`<b>检索诊断</b><span>Precision@3 ${percent(rm.precision_at_k)}</span><span>Recall@3 ${percent(rm.recall_at_k)}</span><span>MRR ${rm.mrr.toFixed(2)}</span>`;
    const gate=data.rag.validation;$('#gate-diagnostic').hidden=false;$('#gate-diagnostic').innerHTML=`<b>证据阀门：${escapeHtml(gate.action)}</b><span>${escapeHtml(gate.reasons.join('；')||'证据包通过验证')}</span><span>类型命中：${escapeHtml(gate.matched_evidence_types.join('、')||'无')}</span>`;
    $('#evidence-list').innerHTML=data.rag.evidence.length?`<h4>证据包</h4>${data.rag.evidence.map(item=>`<a href="${item.url}" target="_blank" rel="noreferrer"><b>[${item.id}] ${escapeHtml(item.title)}</b><span>${escapeHtml(item.organization)} · ${item.year} · ${escapeHtml(item.identifier)} · ${escapeHtml(item.quality)}</span></a>`).join('')}`:'<div class="empty-evidence">证据包为空；REQUIRED 策略下系统应拒绝确定性回答。</div>';
    $('#comparison').hidden=false;
  }catch(error){notice.textContent=`无法运行：${error.message}`;}finally{button.disabled=false;}
}

async function switchDomain(domain){
  if(domain===currentDomain&&$('#question-select').options.length)return;currentDomain=domain;const copy=domainCopy[domain];
  document.querySelectorAll('[data-domain]').forEach(button=>button.classList.toggle('active',button.dataset.domain===domain));
  $('#domain-eyebrow').textContent=copy.eyebrow;$('#domain-title').textContent=copy.title;$('#domain-subtitle').textContent=copy.subtitle;$('#rag-panel-title').textContent=copy.rag;
  $('#summary-grid').innerHTML='';$('#comparison').hidden=true;$('#finding-strip').textContent='正在验证实验管线…';await Promise.all([loadOverview(),loadQuestions()]);await runComparison();
}

async function runFullBenchmark(){
  const button=$('#run-full');button.disabled=true;button.textContent='真实实验运行中…';
  try{const response=await fetch('/api/run-benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:currentDomain,repeats:Number($('#repeat-count').value)})});const data=await responseJson(response);if(!response.ok)throw new Error(errorMessage(data));$('#finding-strip').innerHTML=`<b>真实实验已保存</b><span>${escapeHtml(data.saved_to)} · ${data.model} · ${data.repeats} 次重复</span>`;}catch(error){$('#finding-strip').textContent=`真实实验未运行：${error.message}`;}finally{button.disabled=false;button.textContent='运行真实批量实验';}
}

async function saveModelConfig(event){
  event.preventDefault();const status=$('#model-config-status');status.textContent='正在保存配置…';
  const key=$('#api-key').value.trim();const payload={model:$('#model-select').value,reasoning_effort:$('#reasoning-effort').value};if(key)payload.api_key=key;
  const response=await fetch('/api/model-config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await responseJson(response);$('#api-key').value='';
  if(!response.ok){status.textContent=`保存失败：${errorMessage(data)}`;return;}status.textContent=`已保存到本次服务：${data.model} · reasoning ${data.reasoning_effort}`;await loadStatus();
}

async function testConnection(){
  const button=$('#test-connection'),status=$('#model-config-status');button.disabled=true;status.textContent='正在连接 OpenAI…';
  try{const response=await fetch('/api/model-connection-test',{method:'POST'});const data=await responseJson(response);if(!response.ok)throw new Error(errorMessage(data));status.textContent=`连接成功：${data.model} · ${data.output} · ${data.response_id}`;}catch(error){status.textContent=`连接失败：${error.message}`;}finally{button.disabled=false;}
}

function buildRubricInputs(){
  $('#rubric-inputs').innerHTML=rubricKeys.map(key=>`<label>${metricLabels[key]}<select data-rubric="${key}">${[1,2,3,4,5].map(value=>`<option value="${value}" ${value===3?'selected':''}>${value}</option>`).join('')}</select></label>`).join('');
}

async function saveReview(event){
  event.preventDefault();if(!currentComparison){$('#review-status').textContent='请先运行一次对照';return;}const output=$('#review-arm').value;const answer=currentComparison[displayedOutputs[output]];const payload={domain:currentDomain,question_id:$('#question-select').value,comparison_id:currentComparison.comparison_id,output_code:output,answer_hash:answer.answer_hash,reviewer_alias:$('#reviewer').value,notes:$('#review-notes').value};document.querySelectorAll('[data-rubric]').forEach(input=>payload[input.dataset.rubric]=Number(input.value));
  const response=await fetch('/api/reviews',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();$('#review-status').textContent=response.ok?`已保存，第 ${data.review_count} 条盲评`:`保存失败：${data.detail}`;
}

$('#run').addEventListener('click',runComparison);$('#run-full').addEventListener('click',runFullBenchmark);$('#model-config-form').addEventListener('submit',saveModelConfig);$('#test-connection').addEventListener('click',testConnection);$('#review-form').addEventListener('submit',saveReview);document.querySelectorAll('[data-domain]').forEach(button=>button.addEventListener('click',()=>switchDomain(button.dataset.domain)));document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.view)));
buildRubricInputs();Promise.all([loadStatus(),loadOverview(),loadQuestions()]).then(runComparison).catch(error=>{$('#finding-strip').textContent=`加载失败：${error.message}`;});
