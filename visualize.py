"""HTML visualisation builder for attention and entropy heatmaps."""

import html
import json

import numpy as np

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Attention Heatmap</title>
<style>
  body { margin:0; font-family: Georgia, 'Times New Roman', serif; background:#15171c; }
  .toolbar { position:sticky; top:0; background:#1f2329; color:#e6e6e6; padding:10px 16px;
             display:flex; gap:14px; align-items:center; z-index:10; border-bottom:1px solid #333;
             flex-wrap:wrap; font-size:14px; }
  .toolbar button { background:#333; color:#eee; border:1px solid #555; padding:6px 14px;
                    border-radius:6px; cursor:pointer; font-size:14px; }
  .toolbar button.active { background:#0a84ff; border-color:#0a84ff; }
  .toolbar label { display:flex; align-items:center; gap:6px; }
  .doc { max-width:100ch; margin:24px auto; padding:0 24px 120px; }
  .doc p { line-height:2.1; font-size:18px; text-align:justify; color:#111; background:#fff;
           padding:14px 18px; margin:0 0 16px; border-radius:8px; }
  .w { padding:1px 1px; border-radius:3px; transition: background .12s; cursor:default; }
  .w:hover { outline:2px solid #0a84ff; }
  #tip { position:fixed; background:#000; color:#fff; padding:5px 9px; border-radius:5px;
         font-size:12px; pointer-events:none; display:none; z-index:20; font-family:monospace; }
  .legend { display:flex; align-items:center; gap:8px; }
  .legend .bar { width:160px; height:12px; border-radius:6px; border:1px solid #444; }
  .doc { position:relative; }
  #arcs { position:absolute; inset:0; pointer-events:none; z-index:5; }
  #panel { position:fixed; right:16px; bottom:16px; background:#1f2329; color:#eee;
           padding:10px 12px; border-radius:8px; font-size:13px; max-width:320px; z-index:15;
           display:none; box-shadow:0 4px 16px rgba(0,0,0,.45); font-family:monospace;
           line-height:1.5; }
  #panel b { color:#7fc7ff; }
  .w.target { outline:2px solid #0a84ff; background:#cfe8ff !important; color:#003; }
  .w.bw { background:#fff3b0 !important; color:#332b00; }
  .w.dimmed { color:#c9c9c9 !important; background:transparent !important; }
  body.bw-mode .w { cursor:pointer; }
 </style>
</head>
<body>
<div class="toolbar">
  <button id="b-entropy" class="active">Entropy (per word)</button>
  <button id="b-attention">Attention received (per word)</button>
  <button id="b-backward">Backward attention (click a word)</button>
  <label>Font size <input type="range" id="fs" min="10" max="34" value="18"></label>
  <div class="legend"><span>below median</span><div class="bar" id="legendbar"></div><span>above median</span></div>
</div>
<div class="doc" id="doc"><svg id="arcs" xmlns="http://www.w3.org/2000/svg"></svg>__PARAGRAPHS__</div>
<div id="panel"></div>
<div id="tip"></div>
<script>
const STATS = __STATS__;
const BACKWARD = __BACKWARD__;
function lerp(a,b,t){return a+(b-a)*t;}
function grad(stops,t){
  t=Math.max(0,Math.min(1,t));
  for(let i=0;i<stops.length-1;i++){
    const [p0,c0]=stops[i],[p1,c1]=stops[i+1];
    if(t<=p1){const f=(t-p0)/(p1-p0);
      return `rgb(${lerp(c0[0],c1[0],f)|0},${lerp(c0[1],c1[1],f)|0},${lerp(c0[2],c1[2],f)|0})`;}
  }
  const c=stops[stops.length-1][1];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
// Diverging palette centered on the median: blue (below) -> white (median) -> red (above).
const NEG=[[0,[49,130,189]],[1,[240,240,240]]];
const POS=[[0,[240,240,240]],[1,[215,48,39]]];
const LOW=NEG.map(([p,c])=>`rgb(${c[0]},${c[1]},${c[2]})`).join(',');
const HIGH=POS.map(([p,c])=>`rgb(${c[0]},${c[1]},${c[2]})`).join(',');
function colorFor(t){
  t=Math.max(-1,Math.min(1,t));
  return t<0 ? grad(NEG,t+1) : grad(POS,t);
}
let metric='entropy';
let targetIdx=null;
const doc=document.getElementById('doc');
const svg=document.getElementById('arcs');
const panel=document.getElementById('panel');
const words=[...document.querySelectorAll('.w')];
const SVGNS='http://www.w3.org/2000/svg';
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

function clearArcs(){ while(svg.firstChild) svg.removeChild(svg.firstChild); }

function apply(){
  clearArcs();
  words.forEach(w=>{w.style.background=''; w.classList.remove('dimmed','target','bw');});
  if(metric==='backward'){
    panel.style.display='block';
    panel.innerHTML='<b>Backward attention</b><br>Click a word to draw arcs to the top 10 words it attends to (backwards).';
    return;
  }
  panel.style.display='none';
  const {median,lo,hi}=STATS[metric];
  const maxDev=Math.max(median-lo,hi-median)||1;
  const logMax=Math.log(1+maxDev);
  words.forEach(w=>{
    // Clamp outliers to the IQR fences so extremes don't dominate the scale.
    let v=parseFloat(w.dataset[metric]);
    v=Math.max(lo,Math.min(hi,v));
    const d=v-median;
    const m=logMax>0?Math.log(1+Math.abs(d))/logMax:0;
    w.style.background=colorFor(Math.sign(d)*m);
  });
}
function setActive(){
  ['entropy','attention','backward'].forEach(m=>{
    document.getElementById('b-'+m).classList.toggle('active',metric===m);});
  document.body.classList.toggle('bw-mode',metric==='backward');
}
function pt(w){
  const r=w.getBoundingClientRect();
  const base=svg.getBoundingClientRect();
  return {x:r.left-base.left+r.width/2, y:r.top-base.top+1};
}
function drawArcs(t,links){
  svg.setAttribute('width',doc.scrollWidth);
  svg.setAttribute('height',doc.scrollHeight);
  const vmax=Math.max(...links.map(o=>o.v),1e-9);
  const tp=pt(words[t]);
  links.forEach(o=>{
    const wp=pt(words[o.j]);
    const mx=(tp.x+wp.x)/2, my=(tp.y+wp.y)/2;
    const dist=Math.hypot(tp.x-wp.x,tp.y-wp.y);
    const bow=Math.min(dist*0.45,170);
    const path=document.createElementNS(SVGNS,'path');
    path.setAttribute('d',`M ${tp.x} ${tp.y} Q ${mx} ${my-bow} ${wp.x} ${wp.y}`);
    path.setAttribute('fill','none');
    const s=o.v/vmax;
    path.setAttribute('stroke','#d62728');
    path.setAttribute('stroke-width',(1+8*s).toFixed(2));
    path.setAttribute('stroke-opacity',(0.22+0.63*s).toFixed(2));
    path.setAttribute('stroke-linecap','round');
    svg.appendChild(path);
  });
}
function showBackward(){
  clearArcs();
  words.forEach(w=>{w.style.background=''; w.classList.remove('dimmed','target','bw');});
  const links=(BACKWARD[targetIdx]||[]).map(([j,v])=>({j,v}));
  const involved=new Set([targetIdx,...links.map(o=>o.j)]);
  words.forEach((w,i)=>{ if(!involved.has(i)) w.classList.add('dimmed'); });
  words[targetIdx].classList.add('target');
  links.forEach(o=>words[o.j].classList.add('bw'));
  drawArcs(targetIdx,links);
  const tw=words[targetIdx].textContent;
  if(!links.length){
    panel.innerHTML=`<b>Target:</b> "${esc(tw)}"<br><i>No preceding context.</i>`;
  }else{
    const rows=links.map((o,i)=>`<div>${i+1}. "${esc(words[o.j].textContent)}" &nbsp; ${o.v.toFixed(4)}</div>`).join('');
    panel.innerHTML=`<b>Target:</b> "${esc(tw)}"<br><b>Attends back to:</b>${rows}`;
  }
}
words.forEach(w=>{
  w.addEventListener('click',()=>{
    if(metric!=='backward') return;
    targetIdx=parseInt(w.dataset.idx);
    showBackward();
  });
});
document.getElementById('b-entropy').onclick=()=>{metric='entropy';setActive();apply();};
document.getElementById('b-attention').onclick=()=>{metric='attention';setActive();apply();};
document.getElementById('b-backward').onclick=()=>{metric='backward';setActive();apply();};
function redrawIfBackward(){ if(metric==='backward'&&targetIdx!==null) showBackward(); }
document.getElementById('fs').oninput=e=>{
  document.querySelectorAll('.doc p').forEach(p=>p.style.fontSize=e.target.value+'px');
  redrawIfBackward();};
window.addEventListener('resize',redrawIfBackward);
const tip=document.getElementById('tip');
words.forEach(w=>{
  w.addEventListener('mousemove',ev=>{tip.style.display='block';
    tip.style.left=(ev.clientX+12)+'px'; tip.style.top=(ev.clientY+12)+'px';
    tip.textContent='entropy '+w.dataset.entropy+'   attn '+w.dataset.attention;});
  w.addEventListener('mouseleave',()=>tip.style.display='none');
});
document.getElementById('legendbar').style.background=
  'linear-gradient(90deg,'+LOW+','+HIGH+')';
apply();
</script>
</body>
</html>
"""


def _median(xs: list[float]) -> float:
    """Return the median of *xs*, or ``0.0`` if empty."""
    return float(np.median(xs)) if xs else 0.0


def _iqr_bounds(xs: list[float], k: float = 1.5) -> tuple[float, float]:
    """Compute Tukey fences ``[Q1 - k*IQR, Q3 + k*IQR]`` for outlier filtering.

    Args:
        xs: List of numeric values.
        k: Tukey multiplier (``1.5`` for standard fences, ``3.0`` for extreme).

    Returns:
        ``(lower_fence, upper_fence)`` tuple.
    """
    a = np.asarray(xs, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def build_html(
    word_units: list[list],
    word_entropy: list[float],
    word_attn: list[float],
    backward: list[list[list[float]]],
) -> str:
    """Build the complete standalone HTML document.

    Generates word spans grouped into paragraphs, embeds per-metric statistics
    (median, IQR fences) and the precomputed backward-attention links, then
    substitutes them into :data:`HTML_TEMPLATE`.

    Args:
        word_units: Word units with token indices assigned.
        word_entropy: Per-word entropy values.
        word_attn: Per-word received-attention values.
        backward: Precomputed backward-attention links from
            :func:`metrics.build_backward_attention`.

    Returns:
        A complete HTML document string.
    """
    e_lo, e_hi = _iqr_bounds(word_entropy)
    a_lo, a_hi = _iqr_bounds(word_attn)
    stats = {
        "entropy": {
            "min": float(min(word_entropy)),
            "max": float(max(word_entropy)),
            "median": _median(word_entropy),
            "lo": e_lo,
            "hi": e_hi,
        },
        "attention": {
            "min": float(min(word_attn)),
            "max": float(max(word_attn)),
            "median": _median(word_attn),
            "lo": a_lo,
            "hi": a_hi,
        },
    }
    parts: list[str] = []
    cur: int | None = None
    for i, unit in enumerate(word_units):
        pidx = unit[0]
        wtxt = unit[3]
        if pidx != cur:
            if cur is not None:
                parts.append("</p>")
            parts.append("<p>")
            cur = pidx
        esc = html.escape(wtxt)
        parts.append(
            f'<span class="w" data-idx="{i}" data-entropy="{word_entropy[i]:.5f}" '
            f'data-attention="{word_attn[i]:.6f}">{esc}</span> '
        )
    if cur is not None:
        parts.append("</p>")
    return (
        HTML_TEMPLATE.replace("__PARAGRAPHS__", "".join(parts))
        .replace("__STATS__", json.dumps(stats))
        .replace("__BACKWARD__", json.dumps(backward))
    )
