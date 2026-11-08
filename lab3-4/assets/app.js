/* ДВЕ СТАДИИ. Автономная учебная симуляция. Все вычисления выполняются локально. */
(() => {
'use strict';
const $ = id => document.getElementById(id);
const TRACKS = {
  bio: { code:'Биоинженерия', unit:'с', title:'Наблюдение одной молекулы', description:'Два последовательных молекулярных перехода. Основной канал видит начало A и конечное состояние C.', family:'E', mechanism:'Молекула переходит из состояния A в B, затем из B в C. Если условия постоянны, а вероятность элементарного перехода за короткий промежуток не зависит от уже проведённого времени, каждое ожидание моделируется экспоненциальным законом. Датчик B – дополнительный флуоресцентный канал.' },
  fbit: {code:'ФБИТ', unit:'мс', title:'Доставка события через два спящих узла', description:'Стенд энергосберегающей радиосети. Тревожное сообщение ждёт сеанса приёма сначала узла R1, затем R2. Измеряется время доставки до центра.',family:'U',mechanism:'Охранный датчик отправляет тревожное сообщение через два ретранслятора с duty cycling: каждый периодически открывает короткое окно радиоприёма. Приём сообщения откладывается до ближайшего окна, а время самой передачи пренебрежимо мало. В учебном стенде при каждом новом опыте начальные фазы обоих расписаний заново выбираются независимо и равномерно; для постоянно действующей сети с фиксированными фазами это предположение неверно.'},
  nano: {code:'Физика и наноструктуры', unit:'мкс', title:'Два последовательных туннельных перехода', description:'Носитель заряда проходит два перехода между метастабильными состояниями. Основной датчик фиксирует вход и выход.',family:'E',mechanism:'Носитель находится в одном зарядовом состоянии до случайного туннельного перехода, затем ожидает второго. В стационарной марковской модели каждый переход имеет постоянную интенсивность. Дополнительный зарядовый датчик отмечает промежуточное состояние, не изменяя модель динамики.'}
};
const esc = s => String(s).replace(/[&<>"']/g, m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
const fmt = (n, digits=2) => Number(n).toLocaleString('ru-RU',{minimumFractionDigits:digits,maximumFractionDigits:digits});
const fmtMath = (n, digits=2) => fmt(n,digits).replace(',', '{,}').replaceAll('\u00a0', '\\,');
const pct = p => fmt(100*p,1)+' %';
const clamp = (x,a,b) => Math.min(b,Math.max(a,x));
const pos = x => Math.max(0,x);
const theta = Math.PI*2;
const VARIANT_COUNT = 18;
const query = new URLSearchParams(location.search);
const demo = query.get('demo')==='1'||query.get('mode')==='demo';
const assignedTrack = Object.hasOwn(TRACKS,query.get('track'))?query.get('track'):'bio';
const assignedVariant = /^\d+$/.test(query.get('variant')||'')&&Number(query.get('variant'))>=1&&Number(query.get('variant'))<=VARIANT_COUNT?Number(query.get('variant')):1;
const probabilityInputs = ['decision-q0','decision-q1','decision-q2'];
const numeric = id => {const raw=($(id)?.value||'').trim().replace(/\s/g,'').replace(',','.');if(!raw||!/^[-+]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(raw))return null;const n=Number(raw);return Number.isFinite(n)?n:null};
const mathval=(x,d=3)=>`\\(${fmtMath(x,d)}\\)`;
const mathprob=x=>`\\(${fmtMath(x,4)}\\)`;
const validProb=x=>x!==null&&x>=0&&x<=1;
function clearMath(el){if(el&&window.MathJax&&MathJax.typesetClear)try{MathJax.typesetClear([el])}catch(e){}}
function renderMarkup(id,html){const el=$(id);if(!el||el.dataset.lastmarkup===html)return;clearMath(el);el.innerHTML=html;el.dataset.lastmarkup=html;queueMath();}

// Весь математический набор – MathJax (локальный SVG-вывод, без сетевых зависимостей).
let mathQueued=false;
function setMath(id,latex){
  const el=$(id);
  if(!el)return;
  if(el.dataset.tex===latex)return;
  if(window.MathJax&&MathJax.typesetClear)try{MathJax.typesetClear([el])}catch(e){}
  el.dataset.tex=latex;
  el.textContent='\\('+latex+'\\)';
}
function queueMath(){
  if(mathQueued||!window.MathJax)return;
  mathQueued=true;
  Promise.resolve(MathJax.startup.promise).then(()=>{
    mathQueued=false;
    return MathJax.typesetPromise([document.body]);
  }).catch(e=>{mathQueued=false;console.warn('MathJax:',e)});
}
let state, busy=false, toastTimer;
function seedHash(str){let h=2166136261;for(let i=0;i<str.length;i++){h=Math.imul(h^str.charCodeAt(i),16777619)}return h>>>0}
function random(){let x=state.rng=(state.rng+0x6D2B79F5)>>>0; x=Math.imul(x^(x>>>15),x|1);x^=x+Math.imul(x^(x>>>7),x|61);return ((x^(x>>>14))>>>0)/4294967296}
const BASE_MEANS=[1.00,1.12,1.24,1.06,1.18,0.96,1.10,1.20,1.04,1.28,1.16,1.08];
const RATIO_STARTS={bio:1.78,fbit:1.74,nano:1.79};
const EXTRA_MEANS=[1.14,1.26,1.02,1.22,1.08,1.17];
const EXTRA_RATIOS={
 bio:[1.70,1.73,1.76,2.36,2.39,2.42],
 fbit:[1.60,1.64,1.68,1.72,2.34,2.38],
 nano:[1.71,1.74,1.77,2.37,2.40,2.43]
};
function params(track,variant){
 const original=variant<=12;
 const m=original?BASE_MEANS[variant-1]:EXTRA_MEANS[variant-13];
 const r=original?Number((RATIO_STARTS[track]+0.05*(variant-1)).toFixed(2)):EXTRA_RATIOS[track][variant-13];
 const scale=track==='fbit'?220:track==='nano'?1.9:1;
 return {m1:(variant%2?m*r:m)*scale,m2:(variant%2?m:m*r)*scale};
}
function family(){return TRACKS[state.track].family}
function profile(){return TRACKS[state.track]}
function mu(){return state.params.m1+state.params.m2}
function variance(){const {m1,m2}=state.params;return family()==='E'?m1*m1+m2*m2:(m1*m1+m2*m2)/3}
function nextSample(){const a=random(),b=random();const {m1,m2}=state.params;return family()==='E'? {x:-Math.log(1-a)*m1,y:-Math.log(1-b)*m2}:{x:2*m1*a,y:2*m2*b}}
function expCDF(t,a,b){if(t<=0)return 0;const la=1/a,lb=1/b;if(Math.abs(la-lb)<1e-11)return clamp(1-Math.exp(-la*t)*(1+la*t),0,1);return clamp(1-(lb*Math.exp(-la*t)-la*Math.exp(-lb*t))/(lb-la),0,1)}
function uniformCDF(t,m1,m2){const a=2*m1,b=2*m2;return clamp((pos(t)**2-pos(t-a)**2-pos(t-b)**2+pos(t-a-b)**2)/(2*a*b),0,1)}
function cdfT(t,m1=state.params.m1,m2=state.params.m2){return family()==='E'?expCDF(t,m1,m2):uniformCDF(t,m1,m2)}
function conditional(t){const {m1,m2}=state.params;if(!(t>0))return null;if(family()==='E'){const z=(1/m1-1/m2)*t/2;if(z>700)return 0;if(z< -700)return 1;return 1/(1+Math.exp(z))}const a=Math.max(0,t-2*m2),b=Math.min(t,2*m1);return b<=a?null:clamp((b-Math.max(a,t/2))/(b-a),0,1)}
function normalCDF(z){const t=1/(1+.2316419*Math.abs(z));const d=.3989422804014327*Math.exp(-z*z/2);const p=1-d*t*(.31938153+t*(-.356563782+t*(1.781477937+t*(-1.821255978+t*1.330274429))));return z>=0?p:1-p}
function save(){try{localStorage.setItem('itmo-lr34-research-v1',JSON.stringify(state))}catch(e){}}
function initial(track='bio',variant=1){const p=params(track,variant); const s={version:7,track,variant,params:p,rng:seedHash('itmo-lr34/'+track+'/'+variant),trials:[],sensor:false,prediction:null,predictionNote:'',sensorForecast:'',collapsed:{},modelRevealed:false,decisionForecast:'',decisionLocked:false,decisionVerification:null,decisionInputs:null,decisionAttempt:null,conditionalResponse:null,conditionalAttempts:0,conditionalCorrect:false,conditionalRevealed:false,batchN:100,batchAnswers:null,batchParamsVerified:false,batchParamAttempts:0,batchResults:{},batchReveals:{},batchPredictions:{},batchReveal:false,batchResult:null,threshold:1.15*(p.m1+p.m2),deadline:1.55*(p.m1+p.m2),createdAt:new Date().toISOString()};return s}
function restore(){try{let s=JSON.parse(localStorage.getItem('itmo-lr34-research-v1'));if(s&&(s.version===1||s.version===2||s.version===3||s.version===4||s.version===5||s.version===6||s.version===7)&&TRACKS[s.track]&&s.variant>=1&&s.variant<=VARIANT_COUNT&&Array.isArray(s.trials)){if(s.version<6){state=initial(s.track,s.variant);return false;}const oldRelease=false,oldDecision=false,oldBatchThreshold=s.version===6;s.version=7;s.trials=s.trials.map(r=>r.measured?r:{t:r.t,measured:false});s.collapsed=s.collapsed||{};s.sensorForecast=s.sensorForecast||'';if(oldRelease){s.decisionLocked=false;s.decisionVerification=null;s.decisionChoice=null;s.decisionAttempt=null;s.decisionInputs=null;s.batchResult=null;s.conditionalResponse=null;s.conditionalAttempts=0;s.conditionalCorrect=false;s.conditionalRevealed=false;s.batchParamsVerified=false;s.batchParamAttempts=0;s.batchResults={};s.batchReveals={};s.batchPredictions={};s.collapsed.decision=false;s.collapsed.batch=false;s.collapsed.model=false;}if(oldDecision){s.decisionLocked=false;s.decisionVerification=null;s.decisionAttempt=null;s.decisionInputs=null;s.decisionForecast='';s.batchResults={};s.batchReveals={};s.batchPredictions={};s.collapsed.decision=false;s.collapsed.batch=false;}if(oldBatchThreshold){s.batchResults={};s.batchReveals={};s.batchPredictions={};s.collapsed.batch=false;}state=s;state.decisionInputs=state.decisionInputs||null;state.decisionForecast=state.decisionForecast||'';state.decisionAttempt=state.decisionAttempt||null;state.conditionalResponse=state.conditionalResponse||null;state.batchN=state.batchN||100;state.batchAnswers=state.batchAnswers||null;state.batchReveal=!!state.batchReveal;state.conditionalAttempts=state.conditionalAttempts||0;state.conditionalCorrect=!!state.conditionalCorrect;state.conditionalRevealed=!!state.conditionalRevealed;state.batchParamsVerified=!!state.batchParamsVerified;state.batchParamAttempts=state.batchParamAttempts||0;state.batchResults=state.batchResults||{};state.batchReveals=state.batchReveals||{};state.batchPredictions=state.batchPredictions||{};return true}}catch(e){}state=initial();return false}
function toast(msg){const el=$('toast');el.textContent=msg;el.classList.add('shown');clearTimeout(toastTimer);toastTimer=setTimeout(()=>el.classList.remove('shown'),3500)}
function reset(track=state.track,variant=state.variant,ask=false){if(ask&&!confirm('Удалить текущие измерения и начать новый эксперимент? Экспортируйте журнал заранее, если он нужен.'))return;state=initial(track,variant);busy=false;save();renderAll();toast('Новая экспериментальная сессия готова.')}
function unmeasured(){return state.trials.filter(t=>!t.measured).length}
function measured(){return state.trials.filter(t=>t.measured).length}
function svgApparatus(){const isBio=state.track==='bio',isNano=state.track==='nano', mid=state.sensor;const on=busy?'#ffd09a':'#a7e6d2';let main='';
if(isBio){main=`<g><text x="22" y="23" fill="#9DB9C1" font-size="12" letter-spacing="1">МОЛЕКУЛЯРНЫЕ СОСТОЯНИЯ</text><path d="M68 119H451" stroke="#72939C" stroke-width="2" stroke-dasharray="4 9"/><g fill="#314F59" stroke="#7895A0" stroke-width="1.5"><circle cx="94" cy="119" r="31"/><circle cx="256" cy="119" r="31"/><circle cx="418" cy="119" r="31"/></g><circle cx="94" cy="119" r="14" fill="${on}" opacity=".84"/><circle cx="256" cy="119" r="14" fill="${mid?'#E2B278':'#647E87'}" opacity=".84"/><circle cx="418" cy="119" r="14" fill="#B5CFD3" opacity=".84"/><g fill="#D8E5E6" font-size="17" text-anchor="middle"><text x="94" y="171">A</text><text x="256" y="171">B</text><text x="418" y="171">C</text></g><path d="M133 119H213M296 119H377" stroke="#A6BBC0" stroke-width="1.4"/><g fill="#95B0B7" font-size="11" text-anchor="middle"><text x="175" y="106">переход 1</text><text x="336" y="106">переход 2</text></g></g>`}
else if(isNano){main=`<g><text x="22" y="23" fill="#9DB9C1" font-size="12" letter-spacing="1">ЗАРЯДОВЫЕ СОСТОЯНИЯ</text><path d="M34 170 H73 V68 H139 V170 H213 V46 H285 V170 H361 V76 H437 V170 H486" fill="none" stroke="#7794A1" stroke-width="3"/><g fill="#99DACC"><circle cx="101" cy="160" r="9"/><circle cx="249" cy="160" r="9" opacity="${mid?1:.2}"/><circle cx="399" cy="160" r="9" opacity=".8"/></g><path d="M120 128H227M270 128H394" stroke="#DEAC74" stroke-width="1.7" stroke-dasharray="4 6"/><g fill="#BCD2D5" font-size="12" text-anchor="middle"><text x="100" y="193">A</text><text x="249" y="193">B</text><text x="399" y="193">C</text></g><g fill="#95B0B7" font-size="11"><text x="123" y="118">Туннелирование</text><text x="292" y="118">Туннелирование</text></g></g>`}
else{main=`<g><text x="22" y="23" fill="#9DB9C1" font-size="12" letter-spacing="1">АСИНХРОННАЯ ДОСТАВКА СОБЫТИЯ</text><g stroke="#829EA6" stroke-width="2" fill="none"><path d="M54 129H465"/></g><g fill="#31505B" stroke="#7298A3" stroke-width="1.4"><rect x="24" y="92" width="80" height="70" rx="5"/><rect x="171" y="92" width="82" height="70" rx="5"/><rect x="333" y="92" width="82" height="70" rx="5"/><rect x="462" y="92" width="37" height="70" rx="5"/></g><g font-size="14" fill="#E1EBE9" text-anchor="middle"><text x="64" y="132">A</text><text x="212" y="132">R1</text><text x="374" y="132">R2</text><text x="481" y="132">C</text></g><g stroke="#C5AF88" stroke-width="1.3" fill="none"><path d="M192 82Q212 43 232 82"/><path d="M201 75Q212 57 222 75"/><path d="M354 82Q374 43 394 82"/><path d="M363 75Q374 57 384 75"/></g><g font-size="10" fill="#9DB9C1" text-anchor="middle"><text x="210" y="188">ожидание 1</text><text x="374" y="188">ожидание 2</text></g></g>`}
return `${main}<g><line x1="256" y1="187" x2="256" y2="215" stroke="${mid?'#E2B278':'#506872'}" stroke-dasharray="3 4"/><circle cx="256" cy="218" r="5.5" fill="${mid?'#E2B278':'#3C5962'}" stroke="${mid?'#E2B278':'#879CA3'}"/><text x="271" y="222" font-size="11" fill="${mid?'#E2B278':'#7C949E'}">${mid?'B · измеряется':'B · не регистрируется'}</text></g>`}
function updateStatus(msg, running=false){$('instrument-status').textContent=msg;$('state-light').classList.toggle('pulse',running)}
function runOne(){if(busy)return;busy=true;$('one-trial').disabled=true;$('series').disabled=true;updateStatus('Процесс выполняется',true);const trial=nextSample(), t=trial.x+trial.y;const duration=720; const s=state.sensor; const p=$('action-tip');p.textContent=s?'Идёт регистрация A → B → C. Воспроизведение ускорено.':'Записаны только моменты A и C. Воспроизведение ускорено.';if(s){setTimeout(()=>{updateStatus('Событие B зарегистрировано',true)}, Math.max(90,Math.floor(duration*trial.x/t)))}setTimeout(()=>{state.trials.push(s?{...trial,t,measured:true}:{t,measured:false});busy=false;updateStatus('Проход завершён');save();renderAll();toast(s?'Три временные метки зарегистрированы.':'Полное время зарегистрировано.')},duration)}
function runSeries(){if(busy)return; const s=state.sensor;for(let i=0;i<12;i++){const o=nextSample();state.trials.push(s?{...o,t:o.x+o.y,measured:true}:{t:o.x+o.y,measured:false})}save();renderAll();toast('12 независимых проходов добавлены в журнал.')}
function traceSVG(){let svg='';const all=state.trials.slice(-18), mean=mu(),maxShow=Math.max(3.2*mean,...all.map(t=>t.t))*1.03;const x0=51,x1=646, y0=68,w=x1-x0;const tx=t=>x0+clamp(t/maxShow,0,1)*w;let ticks=5;svg+=`<text x="53" y="26" fill="#C0CED0" font-size="12">${state.trials.length?`Измерения ${Math.max(1,state.trials.length-all.length+1)}–${state.trials.length}`:'Готов к первому измерению'}</text>`;for(let k=0;k<=ticks;k++){let x=x0+w*k/ticks;svg+=`<line x1="${x.toFixed(1)}" y1="${y0-9}" x2="${x.toFixed(1)}" y2="389" stroke="#36515C" stroke-width=".9"/><text x="${x.toFixed(1)}" y="410" fill="#91A8B0" text-anchor="middle" font-size="11">${fmt(maxShow*k/ticks,1)}</text>`}svg+=`<text x="643" y="28" fill="#95B4BA" font-size="10" text-anchor="end">ВРЕМЯ, ${profile().unit}</text>`;
if(!all.length){svg+=`<text x="350" y="222" text-anchor="middle" fill="#D2DEDF" font-family="Georgia,serif" font-size="24">Первый опыт начнёт запись</text><text x="350" y="243" text-anchor="middle" fill="#819CA6" font-size="12">Здесь появятся вход и выход</text>`;return svg}
const h=Math.min(17.4,293/Math.max(all.length,1)); all.forEach((r,i)=>{let y=y0+16+i*h;const xx=tx(r.t),xb=tx(r.x);svg+=`<g><line x1="${x0}" y1="${y}" x2="${xx}" y2="${y}" stroke="${r.measured?'#E8B175':'#748B96'}" stroke-width="3" opacity=".92"/>`;if(r.measured){svg+=`<line x1="${x0}" y1="${y}" x2="${xb}" y2="${y}" stroke="#79C9B8" stroke-width="3.4"/><circle cx="${xb}" cy="${y}" r="3.2" fill="#F0BE78"/>`}svg+=`<circle cx="${x0}" cy="${y}" r="2.3" fill="#9DDFCD"/><circle cx="${xx}" cy="${y}" r="3.3" fill="#E5DBBA"/><text x="11" y="${y+3.5}" fill="#98B1BA" font-size="10">${String(state.trials.length-all.length+i+1).padStart(2,'0')}</text><title>№${state.trials.length-all.length+i+1}: T=${fmt(r.t)} ${profile().unit}${r.measured?`, X=${fmt(r.x)}, Y=${fmt(r.y)}`:', B не измерялось'}</title></g>`});const th=state.threshold,thx=tx(th);svg+=`<line x1="${thx}" y1="51" x2="${thx}" y2="388" stroke="#F3BF83" stroke-width="1.5" stroke-dasharray="4 5"/><text x="${Math.min(x1-3,Math.max(x0+3,thx+5))}" y="46" text-anchor="${thx>x1-70?'end':'start'}" font-size="10" fill="#F3BF83">${fmt(th,1)}</text>`;return svg}
function renderMeasures(){const all=state.trials;const max=3.4*mu();$('threshold').max=String(max);state.threshold=clamp(state.threshold,0,max);$('threshold').value=state.threshold;$('threshold-label').textContent=fmt(state.threshold)+' '+profile().unit;$('threshold-max').textContent=fmt(max,1)+' '+profile().unit;const done=all.filter(v=>v.t<state.threshold).length;$('threshold-share').textContent=all.length?`${done} из ${all.length}  ·  ${pct(done/all.length)}`:'Нет наблюдений';$('traces').innerHTML=traceSVG()}
function renderTrack(){
 const p=profile();
 $('setup-label').textContent=({'bio':'Биоинженерия','fbit':'ФБИТ','nano':'Физика и наноструктуры'}[state.track]);
 $('header-variant-value').textContent=String(state.variant).padStart(2,'0');
 $('header-variant-toggle').setAttribute('aria-label','Выбрать вариант. Текущий: '+String(state.variant).padStart(2,'0'));
 $('track-code').textContent=p.code;
 $('instrument-title').textContent=p.title;$('instrument-description').textContent=p.description;$('unit-label').textContent=p.unit;
 $('apparatus').innerHTML=svgApparatus();$('fbit-phase').hidden=state.track!=='fbit';
 $('total-count').textContent=state.trials.length+' '+(state.trials.length%10===1&&state.trials.length%100!==11?'опыт':(state.trials.length%10>=2&&state.trials.length%10<=4&&(state.trials.length%100<12||state.trials.length%100>14)?'опыта':'опытов'));$('indicator-b').classList.toggle('active',state.sensor);
 $('session-id').textContent='';
 $('series').disabled=busy;$('one-trial').disabled=busy;
 updateStatus(busy?'Процесс выполняется':state.trials.length?'Прибор готов к следующему опыту':'Готово к запуску',busy);
 $('demo-settings-trigger').hidden=!demo;
 document.querySelectorAll('[data-track]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.track===state.track)));
 $('variant-menu-toggle').textContent=String(state.variant).padStart(2,'0')+' ▾';
 document.querySelectorAll('[role="option"][data-variant]').forEach(b=>b.setAttribute('aria-selected',String(Number(b.dataset.variant)===state.variant)));
}
function renderSensor(){
 const checked=document.querySelector(`input[name="hypothesis"][value="${state.prediction||''}"]`);
 document.querySelectorAll('input[name=hypothesis]').forEach(el=>{el.disabled=state.sensor;el.checked=el===checked});
 $('hypothesis-note').disabled=state.sensor;$('hypothesis-note').value=state.predictionNote||'';
 $('sensor-forecast').disabled=state.sensor;$('sensor-forecast').value=state.sensorForecast||'';
 const chosen=state.prediction, can=!state.sensor&&unmeasured()>=6&&Boolean(chosen)&&(state.sensorForecast||'').trim().length>=10;
 $('connect-sensor').disabled=!can;
 $('sensor-hint').textContent=state.sensor?'Вывод и прогноз зафиксированы. Новые запуски регистрируют B.':unmeasured()<6?'Для первого вывода проведите не менее 6 опытов.':!chosen?'Выберите, что позволяют установить текущие измерения.':(state.sensorForecast||'').trim().length<10?'Сформулируйте короткий прогноз о будущих метках B.':'Теперь подключите B: его метки появятся только в новых опытах.';
 $('sensor-result').hidden=!state.sensor;
 if(state.sensor){
  const last=state.trials.filter(t=>t.measured).slice(-1)[0];
  if(last)setMath('latest-x',`${fmtMath(last.x)}\\,\\text{${profile().unit}}`);else $('latest-x').textContent='–';
  if(last)setMath('latest-y',`${fmtMath(last.y)}\\,\\text{${profile().unit}}`);else $('latest-y').textContent='–';
  if(last)setMath('latest-t',`${fmtMath(last.t)}\\,\\text{${profile().unit}}`);else $('latest-t').textContent='–';
  const labels={unknown:'По исходным полным временам определить нельзя',x:'Предположение: больше задерживает первый',y:'Предположение: больше задерживает второй',balanced:'Предположение: вклад обоих примерно одинаков'};
  $('hypothesis-reflection').textContent='Вывод до подключения B: '+(labels[state.prediction]||'–')+'. Прогноз о новых метках: '+(state.sensorForecast||'не записан')+'. Для сравнения эффектов изменения каждой стадии используйте новые измерения и модель.';
 }
}
function modelReady(){return state.sensor&&measured()>=12}
function modelComplete(){return !!(state.modelRevealed&&state.conditionalCorrect)}
function decisionComplete(){return !!(state.decisionLocked&&state.decisionVerification)}
function renderModel(){
 const ready=modelReady();$('model-gate').hidden=ready;$('model-content').hidden=!ready;
 if(!ready)return;
 $('mechanism-text').textContent=profile().mechanism;
 $('reveal-model').hidden=state.modelRevealed;$('model-formulas').hidden=!state.modelRevealed;
 if(!state.modelRevealed)return;
 const {m1,m2}=state.params,p=profile(),isE=family()==='E';
 // Параметры модели набираются тем же экземпляром MathJax.
 const pars=isE?[`\\lambda_1=${fmtMath(1/m1,3)}\\,\\text{${p.unit}}^{-1}`,`\\lambda_2=${fmtMath(1/m2,3)}\\,\\text{${p.unit}}^{-1}`]:[`P_1=${fmtMath(2*m1,2)}\\,\\text{${p.unit}}`,`P_2=${fmtMath(2*m2,2)}\\,\\text{${p.unit}}`];
 const strip=$('parameter-strip');
 const key=pars.join('|');
 if(strip.dataset.key!==key){
   if(window.MathJax&&MathJax.typesetClear)try{MathJax.typesetClear([strip])}catch(e){}
   strip.dataset.key=key;
   strip.innerHTML=`<span>${isE?'Первая интенсивность':'Первый период'} <strong>\\(${pars[0]}\\)</strong></span><span>${isE?'Вторая интенсивность':'Второй период'} <strong>\\(${pars[1]}\\)</strong></span>`;
 }
 setMath('law-line',isE?'X\\sim\\operatorname{Exp}(\\lambda_1),\\quad Y\\sim\\operatorname{Exp}(\\lambda_2)':'X\\sim U(0,P_1),\\quad Y\\sim U(0,P_2)');
 setMath('mean-line',`\\mathbb E[T]`);
 $('law-note').textContent=isE?'Стадии независимы; интенсивности постоянны. Отсутствие памяти – отдельное физическое предположение.':'В каждой сессии фазы приёмников рандомизируются заново и независимо. Для двух узлов с фиксированным расписанием времена могут зависеть.';
 const t=mu();setMath('conditional-t',`t=\\mathbb E[T]=${fmtMath(t,2)}\\,\\text{${p.unit}}`);
 const answer=state.conditionalResponse;
 $('conditional-reveal').hidden=state.conditionalCorrect||state.conditionalAttempts<3||state.conditionalRevealed;
 $('conditional-reveal').disabled=state.conditionalCorrect;
 $('conditional-complete').hidden=!state.conditionalCorrect;
 if(answer){
   if(document.activeElement!==$('conditional-input'))$('conditional-input').value=answer.input;
   if(state.conditionalCorrect){
     renderMarkup('conditional-feedback',`Верно в пределах допуска. Ваш результат: ${mathprob(answer.value)}.${state.conditionalRevealed?' Ответ ранее был раскрыт для самопроверки; это сохранено в журнале.':' Этап модели завершён.'}`);
   }else if(state.conditionalRevealed){
     renderMarkup('conditional-feedback',`Ответ для самопроверки: ${mathprob(conditional(t))}. Он был раскрыт по отдельному запросу; для перехода к прогнозу введите результат и нажмите «Проверить».`);
   }else if(state.conditionalAttempts===1){
     renderMarkup('conditional-feedback','Результат отличается от модели. Проверьте область допустимых значений \\(X\\) при фиксированном \\(T=t\\) и нормировку условной плотности.');
   }else if(state.conditionalAttempts===2){
     renderMarkup('conditional-feedback','Пока не совпадает. Найдите \\(f_{X,T}(x,t)\\), затем разделите её на \\(f_T(t)\\). Интегрируйте только по допустимой части интервала \\(X>t/2\\).');
   }else{
     renderMarkup('conditional-feedback',`Расчёт всё ещё отличается от модели (попыток: ${state.conditionalAttempts}). Сверьте пределы интеграла; при необходимости отдельно откройте ответ для самопроверки.`);
   }
 }else renderMarkup('conditional-feedback','');
 drawScatter();drawCDF();
}
function renderDecision(){
 const ready=modelComplete();$('decision-gate').hidden=ready;$('decision-content').hidden=!ready;if(!ready)return;
 state.deadline=1.55*mu();setMath('deadline-text',`${fmtMath(state.deadline,3)}\\,\\text{${profile().unit}}`);
 const locked=!!state.decisionLocked;
 probabilityInputs.forEach((id,i)=>{const el=$(id);el.disabled=locked;if(state.decisionInputs&&document.activeElement!==el)el.value=String(state.decisionInputs[i]).replace('.',',')});
 $('decision-forecast').disabled=locked;
 if(document.activeElement!==$('decision-forecast'))$('decision-forecast').value=state.decisionForecast||'';
 $('lock-decision').disabled=locked || (state.decisionForecast||'').trim().length<15;
 const vals=state.decisionAttempt;
 if(vals){
  const trueVals=decisionTruth();const errors=vals.map((v,i)=>Math.abs(v-trueVals[i]));
  const good=errors.every(e=>e<=.015);
  renderMarkup('decision-entry-feedback',`${good?'Верно: три риска рассчитаны в пределах допуска 0,015. Прогноз зафиксирован.':'Расчёт пока расходится с моделью. Проверьте закон суммы и область интегрирования; повторная попытка не штрафуется.'} Ваши значения: ${vals.map((v,i)=>`\\(q_${i}=${fmtMath(v,4)}\\)`).join(' · ')}.`);
 }else renderMarkup('decision-entry-feedback','');
 $('decision-results').hidden=!locked;if(!locked)return;
 const q=decisionTruth(),obs=state.decisionVerification;
 const labels=['Исходная система','Изменение модуля I','Изменение модуля II'];
 const entries=q.map((v,i)=>{
  const emp=obs?obs.counts[i]/obs.n:null;
  const d=i===0?'–':mathprob(q[0]-v);
  const empLabel=obs?mathprob(emp):'после серии';
  return `<div class="result-row risk-result-row"><span>${labels[i]}</span><span class="risk-figures"><span title="Ваш прогноз">${mathprob(state.decisionInputs[i])}</span><strong title="Теоретический риск">${mathprob(v)}</strong><span class="risk-empirical" title="Контрольная частота">${empLabel}</span><span title="Абсолютное снижение риска">${d}</span></span></div>`;
 });
 renderMarkup('decision-table',`<div class="risk-table-head"><span>Режим</span><span>Ваш прогноз · Теория · Опыт · Снижение \\(q_0-q_i\\)</span></div>${entries.join('')}`);
 const stronger=q[1]<q[2]?'I':q[2]<q[1]?'II':'одинаков';
 $('decision-explanation').textContent=`Ваш прогноз: «${state.decisionForecast||'–'}». По принятой модели больший абсолютный эффект у изменения ${stronger==='одинаков'?'обоих модулей':`модуля ${stronger}`}. Сравнивайте численные снижения риска, а не только средние времена.`;
 if(obs){
  const freqs=obs.counts.map(c=>c/obs.n),d1=freqs[0]-freqs[1],d2=freqs[0]-freqs[2];
  renderMarkup('improvement-observed',`В 120 сопоставимых новых проходах превышений порога: <strong>${obs.counts[0]} / ${obs.counts[1]} / ${obs.counts[2]}</strong> (исходный режим / изменение I / изменение II). Частоты: ${freqs.map(x=>mathprob(x)).join(' / ')}. Наблюдаемое снижение риска: ${mathprob(d1)} и ${mathprob(d2)}; модель предсказывает ${mathprob(q[0]-q[1])} и ${mathprob(q[0]-q[2])}. При 120 опытах случайный разброс может изменить даже порядок наблюдаемых снижений – аналитические риски остаются главным прогнозом.`);
  $('run-improvement').disabled=true;
 }else{renderMarkup('improvement-observed','Сначала зафиксирован расчёт и словесный прогноз; теперь можно получить независимую контрольную серию.');$('run-improvement').disabled=false;}
}
function decisionTruth(){const {m1,m2}=state.params,t=state.deadline;return [1-cdfT(t,m1,m2),1-cdfT(t,.75*m1,m2),1-cdfT(t,m1,.75*m2)]}
function renderBatch(){
 const ready=decisionComplete();$('batch-gate').hidden=ready;$('batch-content').hidden=!ready;if(!ready)return;
 const n=state.batchN||100;
 document.querySelectorAll('[data-n]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.n)===n)));
 const limit=1.04*mu();setMath('batch-limit',`${fmtMath(limit,3)}\\,\\text{${profile().unit}}`);$('batch-axis-unit').textContent=profile().unit;
 const done=!!state.batchParamsVerified;
 $('batch-mu').disabled=done;$('batch-sigma').disabled=done;
 $('batch-check-params').hidden=done;
 $('batch-param-done').hidden=!done;
 $('batch-forecast-zone').hidden=!done;
 if(state.batchAnswers){
   if(document.activeElement!==$('batch-mu'))$('batch-mu').value=String(state.batchAnswers[0]??'').replace('.',',');
   if(document.activeElement!==$('batch-sigma'))$('batch-sigma').value=String(state.batchAnswers[1]??'').replace('.',',');
 }
 if(done)renderMarkup('batch-param-feedback','Характеристики модели проверены; теперь зафиксируйте прогноз для выбранного размера партии.');
 const result=state.batchResults[String(n)]||null;
 $('batch-chart-section').hidden=!result;
 $('batch-reveal').hidden=!result;
 $('batch-reveal').textContent=state.batchReveals[String(n)]?'Скрыть приближение ЦПТ':'Показать правильное приближение ЦПТ';
 $('batch-prob').disabled=!!result;
 $('simulate-batches').disabled=!!result;
 if(result){
   if(document.activeElement!==$('batch-prob'))$('batch-prob').value=String(result.user[2]).replace('.',',');
   const observed=result.means.filter(x=>x<=result.limit).length/result.means.length;
   renderMarkup('batch-results',`Для \(n=${n}\): ваш прогноз – ${mathprob(result.user[2])}, моделирование 300 партий – ${mathprob(observed)}. Это сравнение прогноза с выборкой, не точная проверка равенства.`+
     (n===100?' <strong>Обязательная контрольная партия выполнена.</strong>':' Это исследовательский режим; для завершения ЛР выполните также обязательное \(n=100\).'));
   if(state.batchReveals[String(n)]){
     renderMarkup('batch-correct',`Нормальное приближение ЦПТ для \(n=${n}\): \(P_{\mathrm{ЦПТ}}=${fmtMath(result.approx,4)}\). Различие с моделированием связано с конечным числом партий и приближённостью ЦПТ.`);
     $('batch-correct').hidden=false;
   }else $('batch-correct').hidden=true;
   drawBatch(result);
 }else{
   if(document.activeElement!==$('batch-prob'))$('batch-prob').value=state.batchPredictions[String(n)]||'';
   renderMarkup('batch-results',done?`Для \(n=${n}\) введите и зафиксируйте свой прогноз до моделирования.`:'Сначала проверьте аналитические значения среднего и стандартного отклонения.');
   $('batch-correct').hidden=true;
 }
 $('batch-hint').textContent=n===100?'100 – обязательная контрольная партия: 300 независимых повторений.':'5 и 20 – дополнительные наблюдения разброса; обязательная партия – 100.';
}
function renderAll(){renderTrack();renderMeasures();renderSensor();renderModel();renderDecision();renderBatch();setSteps();renderFoldButtons();queueMath();save()}
function stepStatuses(){return {experiment:state.sensor,sensor:modelReady(),model:modelComplete(),decision:decisionComplete(),batch:!!state.batchResults['100']}}
function setSteps(){
 const completed=stepStatuses();const opened={experiment:true,sensor:unmeasured()>=1||state.sensor,model:modelReady(),decision:modelComplete(),batch:decisionComplete()};
 let active=decisionComplete()?'batch':modelComplete()?'decision':modelReady()?'model':state.sensor?'sensor':'experiment';
 document.querySelectorAll('.step').forEach(n=>{const id=n.dataset.step;n.classList.toggle('active',id===active);n.classList.toggle('unavailable',!opened[id]);n.classList.toggle('complete',!!completed[id]);n.setAttribute('aria-disabled',String(!opened[id]));});
}
function foldSummaries(){
 const meta={
  experiment:`${state.trials.length} опытов · ${unmeasured()} до датчика, ${measured()} после`,
  sensor:state.sensor?'Промежуточный датчик подключён; записи A–B–C доступны для новых проходов.':'',
  model:state.modelRevealed?`Модель ${family()} · ${state.conditionalCorrect?'условная вероятность проверена':'ожидается условный расчёт'}${state.conditionalRevealed?' · ответ раскрывался':''}`:'',
  decision:state.decisionLocked?`Сравнены два вмешательства; ${state.decisionVerification?'контрольная серия 120 проведена':'прогноз зафиксирован'}`:'',
  batch:state.batchResults['100']?'100 объектов × 300 партий · обязательная проверка завершена':'',
 };
 for(const [id,summary] of Object.entries(meta)){
   const el=$('summary-'+id);
   if(el){el.textContent=summary;el.hidden=!(state.collapsed&&state.collapsed[id]);}
 }
}
function renderFoldButtons(){
 const done=stepStatuses();
 document.querySelectorAll('[data-fold]').forEach(btn=>{
  const id=btn.dataset.fold;btn.hidden=!done[id];
  const region=id==='experiment'?document.querySelector('.instrument'):$(id);
  const folded=!!(state.collapsed&&state.collapsed[id]);
  region.classList.toggle('is-collapsed',folded);
  btn.textContent=folded?'Развернуть этап ↓':'Свернуть этап ↑';
  btn.setAttribute('aria-expanded',String(!folded));
 });
 foldSummaries();
}
// Canvas использует CSS-координаты и физическое разрешение devicePixelRatio.
// При ресайзе размеры пересчитываются; текст и графики остаются резкими на HiDPI.
function canvasFrame(el){
 const cssW=el.getBoundingClientRect().width||el.clientWidth||560;
 const w=Math.max(120,Math.round(cssW));
 const wide=el.id==='batch-plot';
 const h=wide?Math.max(210,Math.min(330,Math.round(w*.37))):Math.max(236,Math.min(325,Math.round(w*.68)));
 const dpr=Math.min(3,Math.max(1,window.devicePixelRatio||1));
 if(el.width!==Math.round(w*dpr)||el.height!==Math.round(h*dpr)){
  el.style.height=h+'px';el.width=Math.round(w*dpr);el.height=Math.round(h*dpr);
 }
 const c=el.getContext('2d');c.setTransform(dpr,0,0,dpr,0,0);
 c.clearRect(0,0,w,h);c.fillStyle='#fffefa';c.fillRect(0,0,w,h);
 const L=51,R=w-18,T=24,B=h-39;
 c.strokeStyle='#c9d4d0';c.lineWidth=1;c.beginPath();
 c.moveTo(L+.5,T);c.lineTo(L+.5,B+.5);c.lineTo(R,B+.5);c.stroke();
 return {c,w,h,L,R,T,B,dpr};
}
function plotText(q,text,x,y,color='#687d7c',size=11,align='left'){
 const c=q.c;c.fillStyle=color;c.font=`${size}px system-ui, -apple-system, sans-serif`;c.textAlign=align;c.textBaseline='middle';c.fillText(text,x,y);c.textAlign='left';
}
function tickStep(maxv,target=4){
 const raw=Math.max(1e-6,maxv/target),p=Math.pow(10,Math.floor(Math.log10(raw))),scaled=raw/p;
 return (scaled<1.5?1:scaled<3.5?2:scaled<7.5?5:10)*p;
}
function plotAxes(q,maxX,maxY=1,yLabel='Доля'){
 const {c,L,R,T,B}=q;
 for(let j=0;j<=4;j++){
  const yy=B-j*(B-T)/4,v=maxY*j/4;
  c.lineWidth=.75;c.strokeStyle='#e4eae7';c.beginPath();c.moveTo(L,yy+.5);c.lineTo(R,yy+.5);c.stroke();
  plotText(q,fmt(v,maxY<=2?2:1),L-7,yy,'#839391',10,'right');
 }
 const step=tickStep(maxX,4),digits=step<1?2:step<10?1:0;
 for(let x=0;x<=maxX+step*.01;x+=step){
  const xx=L+(R-L)*(x/maxX);
  c.strokeStyle='#edf1ee';c.lineWidth=.75;c.beginPath();c.moveTo(xx+.5,T);c.lineTo(xx+.5,B);c.stroke();
  plotText(q,fmt(x,digits),xx,B+15,'#82938d',10,'center');
 }
 plotText(q,yLabel,L,T-10,'#5e777a',10);
}
function clearCanvas(el,msg){
 const q=canvasFrame(el);plotText(q,msg,q.w/2,q.h/2,'#82938c',12,'center');
}
function drawScatter(){
 const q=canvasFrame($('scatter')),{c,L,R,T,B}=q;
 const points=state.trials.filter(p=>p.measured),mx=state.params.m1,my=state.params.m2;
 const maxX=family()==='E'?Math.max(3*mx,...points.map(p=>p.x))*1.08:2*mx*1.04;
 const maxY=family()==='E'?Math.max(3*my,...points.map(p=>p.y))*1.08:2*my*1.04;
 plotAxes(q,maxX,maxY,'Y · '+profile().unit);
 // Область определения совместной плотности семейства U.
 if(family()==='U'&&state.decisionLocked){
   c.strokeStyle='#a8c5bf';c.lineWidth=1.2;c.setLineDash([4,4]);
   c.strokeRect(L,B-2*my/maxY*(B-T),2*mx/maxX*(R-L),2*my/maxY*(B-T));c.setLineDash([]);
 }
 points.forEach((p)=>{
  const x=L+p.x/maxX*(R-L),y=B-p.y/maxY*(B-T);
  c.fillStyle='rgba(47,122,116,.57)';c.beginPath();c.arc(x,y,2.9,0,theta);c.fill();
 });
 plotText(q,'X · '+profile().unit,R,B+31,'#597572',11,'right');
 if(!points.length)plotText(q,'Промежуточные измерения ещё не получены',q.w/2,q.h/2,'#859994',11,'center');
}
function drawCDF(){
 const q=canvasFrame($('cdf')),{c,L,R,T,B}=q,all=state.trials;
 const xmax=family()==='U'?2*(state.params.m1+state.params.m2)*1.06:3.5*mu();
 const x=t=>L+clamp(t/xmax,0,1)*(R-L),y=v=>B-clamp(v,0,1)*(B-T);
 plotAxes(q,xmax,1,'F');
 if(state.decisionLocked){
  c.beginPath();for(let j=0;j<=240;j++){const t=j/240*xmax;j?c.lineTo(x(t),y(cdfT(t))):c.moveTo(x(t),y(cdfT(t)))}
  c.strokeStyle='#2c847e';c.lineWidth=2.5;c.lineJoin='round';c.stroke();
 }
 if(all.length){
  const sorted=all.map(v=>v.t).sort((a,b)=>a-b), n=sorted.length;
  c.beginPath();c.moveTo(x(0),y(0));let old=0;
  sorted.forEach((t,i)=>{const xx=x(t);c.lineTo(xx,y(i/n));c.lineTo(xx,y((i+1)/n));old=(i+1)/n});
  c.lineTo(R,y(old));c.strokeStyle='#bc8a59';c.lineWidth=1.65;c.lineJoin='miter';c.stroke();
 }
 if(state.decisionLocked)plotText(q,'Теория / CDF',L+10,T+14,'#24776d',10);
 plotText(q,'Эмпирическая частота',L+(state.decisionLocked?108:10),T+14,'#a9774e',10);
 plotText(q,profile().unit,R,B+31,'#637b78',10,'right');
}
function drawBatch(r){
 const q=canvasFrame($('batch-plot')),{c,L,R,T,B}=q;
 const n=r.n,sigma=Math.sqrt(variance()/n),center=mu(),left=Math.max(0,center-4*sigma),right=center+4*sigma;
 const buckets=q.w<420?13:q.w<720?19:23,step=(right-left)/buckets,counts=new Array(buckets).fill(0);
 r.means.forEach(v=>{const idx=Math.floor((v-left)/step);if(idx>=0&&idx<buckets)counts[idx]++});
 const normal=x=>Math.exp(-.5*((x-center)/sigma)**2)/(sigma*Math.sqrt(2*Math.PI));
 const theoreticalMax=normal(center)*step*r.means.length;
 const scale=Math.max(...counts,theoreticalMax)*1.22;
 const y=v=>B-v/scale*(B-T);
 for(let j=0;j<=4;j++){
  const yy=B-j*(B-T)/4;
  c.strokeStyle='#e8eeea';c.lineWidth=.75;c.beginPath();c.moveTo(L,yy);c.lineTo(R,yy);c.stroke();
  plotText(q,String(Math.round(scale*j/4)),L-8,yy,'#82928f',10,'right');
 }
 for(let i=0;i<buckets;i++){
  const xx=L+i*(R-L)/buckets+1.6,w=(R-L)/buckets-3.2;
  c.fillStyle='rgba(69,146,134,.57)';c.fillRect(xx,y(counts[i]),w,B-y(counts[i]));
 }
 c.beginPath();for(let i=0;i<=230;i++){
  const x=left+(right-left)*i/230,xx=L+(R-L)*i/230,yy=y(normal(x)*step*r.means.length);
  i?c.lineTo(xx,yy):c.moveTo(xx,yy);
 }
 c.strokeStyle='#b17c52';c.lineWidth=2.4;c.lineJoin='round';c.stroke();
 for(const [x,align] of [[left,'left'],[center,'center'],[right,'right']]){
  plotText(q,fmt(x,1),L+(x-left)/(right-left)*(R-L),B+17,'#697b76',10,align);
 }
 plotText(q,'300 независимых партий · частота по интервалам',L+10,T+13,'#536d69',11);
 plotText(q,profile().unit,R,B+31,'#637b78',10,'right');
}
function collectExport(){return {schema:'itmo-lr34-two-stages/v3',exportedAt:new Date().toISOString(),track:state.track,variant:state.variant,unit:profile().unit,modelFamily:family(),revealedParameters:state.modelRevealed?state.params:null,experiment:{initialDataOnlyTotalTime:true,sensorEnabled:state.sensor,hypothesis:state.prediction,hypothesisNote:state.predictionNote,forecastForSensor:state.sensorForecast||'',measurements:state.trials.map((r,i)=>({trial:i+1,intermediateMeasured:r.measured,T:r.t,X:r.measured?r.x:null,Y:r.measured?r.y:null})),threshold:state.threshold,deadline:state.deadline},decision:{studentForecast:state.decisionForecast,studentProbabilities:state.decisionInputs,locked:state.decisionLocked,comparisonOfBothInterventions:true,verification:state.decisionVerification},conditionalSelfcheck:{attempts:state.conditionalAttempts,correct:state.conditionalCorrect,revealed:state.conditionalRevealed},batches:{analyticalParametersVerified:state.batchParamsVerified,results:state.batchResults,disclosedNormalAnswers:state.batchReveals},modelingNotice:'Это псевдослучайная учебная модель; физические измерения не выполнялись.'}}
function download(name,text,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2000)}
function initEvents(){
 $('demo-settings-trigger').addEventListener('click',()=>{const box=$('demo-settings'),next=box.hidden;box.hidden=!next;$('demo-settings-trigger').setAttribute('aria-expanded',String(next));});
 document.querySelectorAll('[data-track]').forEach(b=>b.addEventListener('click',()=>{if(!demo||b.dataset.track===state.track)return;if(confirm('Сменить направление и начать новую сессию?'))reset(b.dataset.track,state.variant)}));
 const menu=$('variant-menu');for(let i=1;i<=VARIANT_COUNT;i++){const b=document.createElement('button');b.type='button';b.setAttribute('role','option');b.tabIndex=-1;b.dataset.variant=i;b.textContent=String(i).padStart(2,'0');menu.append(b);b.addEventListener('click',()=>chooseVariant(i));}
 function closeMenu(){menu.hidden=true;$('variant-menu-toggle').setAttribute('aria-expanded','false');}
 function chooseVariant(i){if(!demo)return;closeMenu();if(i===state.variant){$('variant-menu-toggle').focus();return}if(confirm('Сменить вариант и сбросить измерения?'))reset(state.track,i);$('variant-menu-toggle').focus()}
 $('variant-prev').addEventListener('click',()=>chooseVariant(state.variant===1?VARIANT_COUNT:state.variant-1));
 $('variant-next').addEventListener('click',()=>chooseVariant(state.variant===VARIANT_COUNT?1:state.variant+1));
 $('variant-menu-toggle').addEventListener('click',()=>{menu.hidden=!menu.hidden;const opened=!menu.hidden;$('variant-menu-toggle').setAttribute('aria-expanded',String(opened));if(opened){const chosen=menu.querySelector(`[data-variant="${state.variant}"]`);chosen?.focus();}});
 menu.addEventListener('keydown',e=>{const items=[...menu.querySelectorAll('[role=option]')];const k=items.indexOf(document.activeElement);if(e.key==='Escape'){closeMenu();$('variant-menu-toggle').focus();e.preventDefault()}else if(['ArrowDown','ArrowRight','ArrowUp','ArrowLeft','Home','End'].includes(e.key)){e.preventDefault();const idx=e.key==='Home'?0:e.key==='End'?items.length-1:(k+(e.key==='ArrowDown'||e.key==='ArrowRight'?1:-1)+items.length)%items.length;items[idx].focus();}else if(e.key==='Enter'||e.key===' '){document.activeElement?.click();e.preventDefault()}});
 document.addEventListener('click',e=>{if(!e.target.closest('.variant-nav'))closeMenu()});

 const headToggle=$('header-variant-toggle'), headMenu=$('header-variant-menu'),headOptions=$('header-variant-options');
 const closeHead=()=>{headMenu.hidden=true;headToggle.setAttribute('aria-expanded','false')};
 for(let i=1;i<=VARIANT_COUNT;i++){
  const item=document.createElement('button');item.type='button';item.role='option';item.tabIndex=-1;
  item.dataset.variant=i;item.textContent=String(i).padStart(2,'0');
  item.setAttribute('aria-label','Вариант '+String(i).padStart(2,'0'));
  item.addEventListener('click',()=>{
   closeHead();if(i!==state.variant&&(!state.trials.length||confirm('Сменить вариант? Текущие измерения будут сброшены.')))reset(state.track,i);
   headToggle.focus();
  });headOptions.append(item);
 }
 headToggle.addEventListener('click',()=>{
  const open=headMenu.hidden;headMenu.hidden=!open;headToggle.setAttribute('aria-expanded',String(open));
  if(open){closeMenu();headOptions.querySelector(`[data-variant="${state.variant}"]`)?.focus()}
 });
 headOptions.addEventListener('keydown',e=>{
  const items=[...headOptions.querySelectorAll('[role="option"]')],current=items.indexOf(document.activeElement);
  if(e.key==='Escape'){closeHead();headToggle.focus();e.preventDefault()}
  else if(['ArrowDown','ArrowRight','ArrowUp','ArrowLeft','Home','End'].includes(e.key)){
   const forward=e.key==='ArrowRight'||e.key==='ArrowDown';
   const ix=e.key==='Home'?0:e.key==='End'?items.length-1:(current+(forward?1:-1)+items.length)%items.length;
   items[ix]?.focus();e.preventDefault();
  }
 });
 headToggle.addEventListener('keydown',e=>{if(e.key==='ArrowDown'||e.key==='ArrowUp'){
  e.preventDefault();headMenu.hidden=false;headToggle.setAttribute('aria-expanded','true');
  headOptions.querySelector(`[data-variant="${state.variant}"]`)?.focus();
 }});
 document.addEventListener('click',e=>{if(!e.target.closest('.header-variant'))closeHead()});
 document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!headMenu.hidden){closeHead();headToggle.focus()}});
 $('reset').addEventListener('click',()=>reset(state.track,state.variant,true));$('student-reset').addEventListener('click',()=>reset(state.track,state.variant,true));
 $('one-trial').addEventListener('click',runOne);$('series').addEventListener('click',runSeries);
 $('threshold').addEventListener('input',e=>{state.threshold=Number(e.target.value);renderMeasures();save()});
 document.querySelectorAll('input[name=hypothesis]').forEach(el=>el.addEventListener('change',e=>{state.prediction=e.target.value;renderSensor();save()}));
 $('hypothesis-note').addEventListener('input',e=>{state.predictionNote=e.target.value;save()});
 $('sensor-forecast').addEventListener('input',e=>{state.sensorForecast=e.target.value;renderSensor();save()});
 document.querySelectorAll('[data-fold]').forEach(btn=>btn.addEventListener('click',()=>{const id=btn.dataset.fold;state.collapsed=state.collapsed||{};state.collapsed[id]=!state.collapsed[id];renderFoldButtons();requestAnimationFrame(()=>{if(state.modelRevealed){drawScatter();drawCDF()}const br=state.batchResults[String(state.batchN)];if(br)drawBatch(br)});save();}));
 $('connect-sensor').addEventListener('click',()=>{if(state.sensor||unmeasured()<6||!state.prediction||(state.sensorForecast||'').trim().length<10)return;state.sensor=true;renderAll();toast('Датчик B включён. Новые опыты будут иметь три метки.')});
 $('reveal-model').addEventListener('click',()=>{if(!modelReady())return;state.modelRevealed=true;state.collapsed.experiment=true;state.collapsed.sensor=true;renderAll();toast('Характеристики установки открыты. Параметры не оценивались по выборке.')});
 $('conditional-check').addEventListener('click',()=>{
  const val=numeric('conditional-input');if(!validProb(val)){renderMarkup('conditional-feedback','Введите вероятность от 0 до 1.');return}
  if(state.conditionalCorrect)return;
  state.conditionalAttempts++;
  state.conditionalResponse={input:$('conditional-input').value,value:val};
  state.conditionalCorrect=Math.abs(val-conditional(mu()))<=.015;
  if(state.conditionalCorrect)state.collapsed.sensor=true;
  renderAll();
 });
 $('conditional-reveal').addEventListener('click',()=>{if(state.conditionalAttempts<3||state.conditionalCorrect)return;state.conditionalRevealed=true;renderAll()});
 probabilityInputs.forEach((id,i)=>$(id).addEventListener('input',()=>{state.decisionAttempt=null;if(!state.decisionLocked){state.decisionInputs=probabilityInputs.map(x=>$(x).value);renderMarkup('decision-entry-feedback','');}save()}));
 $('decision-forecast').addEventListener('input',e=>{if(state.decisionLocked)return;state.decisionForecast=e.target.value;renderDecision();save()});
 $('lock-decision').addEventListener('click',()=>{
  if(!modelComplete()||(state.decisionForecast||'').trim().length<15)return;
  const values=probabilityInputs.map(numeric);
  if(!values.every(validProb)){renderMarkup('decision-entry-feedback','Заполните все три поля вероятностями от 0 до 1.');return}
  state.decisionInputs=values;state.decisionAttempt=values.slice();
  const truth=decisionTruth();const valid=values.every((v,i)=>Math.abs(v-truth[i])<=.015);
  if(valid){state.decisionLocked=true;state.collapsed.model=true;toast('Расчёты проверены. Оба вмешательства можно сравнить на новой серии.')}else toast('Результаты различаются; исправьте значения и повторите проверку.');
  renderDecision();renderFoldButtons();drawScatter();drawCDF();queueMath();save();
 });
 $('run-improvement').addEventListener('click',()=>{
  if(!state.decisionLocked||state.decisionVerification)return;
  const counts=[0,0,0],limit=state.deadline;
  for(let i=0;i<120;i++){
   const {x,y}=nextSample(),values=[x+y,.75*x+y,x+.75*y];
   values.forEach((t,j)=>{if(t>limit)counts[j]++});
  }
  // Одна и та же случайная пара используется в трёх сценариях только для снижения шума сравнения.
  state.decisionVerification={counts,n:120,paired:true};renderAll();
 });
 document.querySelectorAll('[data-n]').forEach(b=>b.addEventListener('click',()=>{
   state.batchN=Number(b.dataset.n);renderBatch();queueMath();save();
 }));
 ['batch-mu','batch-sigma'].forEach((id,i)=>$(id).addEventListener('input',()=>{
   if(state.batchParamsVerified)return;
   state.batchAnswers=state.batchAnswers||['','',''];state.batchAnswers[i]=$(id).value;
   renderMarkup('batch-param-feedback','');save();
 }));
 $('batch-check-params').addEventListener('click',()=>{
   if(!decisionComplete()||state.batchParamsVerified)return;
   const a=numeric('batch-mu'),b=numeric('batch-sigma');
   if(a===null||b===null||a<=0||b<=0){renderMarkup('batch-param-feedback','Введите положительные численные значения среднего и стандартного отклонения.');return}
   state.batchParamAttempts++;
   state.batchAnswers=[a,b,''];
   const muOK=Math.abs(a-mu())<=Math.max(.015*mu(),.0001);
   const sigma=Math.sqrt(variance());const sigmaOK=Math.abs(b-sigma)<=Math.max(.02*sigma,.0001);
   if(muOK&&sigmaOK){state.batchParamsVerified=true;toast('Характеристики модели проверены. Теперь зафиксируйте прогноз.');}
   else renderMarkup('batch-param-feedback',state.batchParamAttempts===1?'Параметры не совпадают. Пересчитайте математические ожидания и дисперсии двух стадий; при независимости дисперсии складываются.':'Параметры пока не совпадают. Проверьте, что введено стандартное отклонение \(\sigma=\sqrt{\operatorname{Var}(T)}\), а не дисперсия; единицы измерения тоже важны.');
   renderAll();
 });
 $('batch-prob').addEventListener('input',()=>{
   if(state.batchResults[String(state.batchN)])return;
   state.batchPredictions[String(state.batchN)]=$('batch-prob').value;save();
 });
 $('simulate-batches').addEventListener('click',()=>{
   if(!decisionComplete()||!state.batchParamsVerified)return;
   const n=state.batchN;if(state.batchResults[String(n)])return;
   const predicted=numeric('batch-prob');
   if(!validProb(predicted)){renderMarkup('batch-forecast-feedback','Введите прогноз вероятности от 0 до 1.');return}
   const means=[];for(let j=0;j<300;j++){let sum=0;for(let k=0;k<n;k++){const smp=nextSample();sum+=smp.x+smp.y}means.push(sum/n)}
   const limit=1.04*mu(),approx=normalCDF((limit-mu())/Math.sqrt(variance()/n));
   renderMarkup('batch-forecast-feedback','');state.batchPredictions[String(n)]=$('batch-prob').value;
   state.batchResults[String(n)]={n,means,limit,approx,user:[state.batchAnswers[0],state.batchAnswers[1],predicted]};
   if(n===100)state.collapsed.decision=true;
   renderAll();
 });
 $('batch-reveal').addEventListener('click',()=>{
   const key=String(state.batchN);if(!state.batchResults[key])return;
   state.batchReveals[key]=!state.batchReveals[key];renderBatch();queueMath();save();
 });
 $('export-json').addEventListener('click',()=>{download(`LR34_${state.track}_variant-${state.variant}.json`,JSON.stringify(collectExport(),null,2),'application/json;charset=utf-8');toast('Журнал выгружен в JSON.')});
 $('export-csv').addEventListener('click',()=>{const rows=[['trial','sensor_B_measured','X','Y','T','unit'],...state.trials.map((v,i)=>[i+1,v.measured?'yes':'no',v.measured?v.x.toFixed(6):'',v.measured?v.y.toFixed(6):'',v.t.toFixed(6),profile().unit])];download(`LR34_${state.track}_variant-${state.variant}.csv`,rows.map(row=>row.join(',')).join('\n'),'text/csv;charset=utf-8');toast('Измерения выгружены в CSV.')});
}
let resizeQueued=false;
window.addEventListener('resize',()=>{
 if(resizeQueued)return;resizeQueued=true;
 requestAnimationFrame(()=>{
   resizeQueued=false;
   if(state.modelRevealed){drawScatter();drawCDF()}
   const result=state.batchResults[String(state.batchN)];if(result)drawBatch(result);
 });
});
restore();
if(state.track!==assignedTrack||state.variant!==assignedVariant)state=initial(assignedTrack,assignedVariant);
initEvents();renderAll();
})();
