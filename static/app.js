'use strict';
let studentId = localStorage.getItem('offline_classroom_student_id') || '';
let lastSlide = -1;

const $ = id => document.getElementById(id);
async function api(path, body) {
  const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)} : {cache:'no-store'});
  return r.json();
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));
}

async function refreshStudent() {
  const s = await api('/api/state');
  if (!$('classView')) return;
  $('lessonTitle').textContent = s.lesson.title;
  $('studentCount').textContent = s.student_count + ' students connected';
  if (s.slide !== lastSlide) { $('lessonContent').innerHTML = s.lesson.content; lastSlide=s.slide; }
  renderStudentPoll(s);
}
function renderStudentPoll(s) {
  const box=$('pollBox'); if(!box) return;
  if(!s.poll){box.innerHTML=''; return;}
  let html='<div class="poll"><h2>Answer</h2><p>'+escapeHtml(s.poll.question)+'</p><div class="answers">';
  s.poll.options.forEach((o,i)=>{html+='<button class="answer" data-answer="'+i+'">'+escapeHtml(o)+'</button>';});
  html+='</div></div>'; box.innerHTML=html;
  box.querySelectorAll('.answer').forEach(b=>b.addEventListener('click', async ()=>{await api('/api/answer',{id:studentId,answer:b.dataset.answer}); box.innerHTML='<div class="card"><h2>Answer submitted</h2><p>Wait for the teacher.</p></div>';}));
}
async function joinClass() {
  const name=$('studentName').value.trim(); if(!name){$('joinMsg').textContent='Please enter your name.'; return;}
  const x=await api('/api/join',{name}); if(!x.ok){$('joinMsg').textContent=x.error||'Could not join.'; return;}
  studentId=x.id; localStorage.setItem('offline_classroom_student_id',studentId); $('joinView').hidden=true; $('classView').hidden=false; refreshStudent();
}

async function refreshTeacher() {
  const s=await api('/api/state'); if(!$('teacherContent')) return;
  $('teacherTitle').textContent=(s.slide+1)+' / '+s.total_slides+' — '+s.lesson.title;
  $('teacherCount').textContent=s.student_count+' students connected';
  $('teacherContent').innerHTML=s.lesson.content;
  if(!s.poll){$('results').textContent='No poll active.'; return;}
  const totals=s.poll.options.map((_,i)=>Object.values(s.answers).filter(v=>v===String(i)).length);
  let h='<p><b>'+escapeHtml(s.poll.question)+'</b></p>';
  s.poll.options.forEach((o,i)=>{h+='<div class="result"><span>'+escapeHtml(o)+'</span><b>'+totals[i]+'</b></div>';});
  h+='<p>'+Object.keys(s.answers).length+' / '+s.student_count+' answered</p>';
  $('results').innerHTML=h;
}
async function teacherPost(path, body){await api(path,body); refreshTeacher();}
function setup() {
  if($('joinBtn')) $('joinBtn').addEventListener('click',joinClass);
  if($('studentName')) $('studentName').addEventListener('keydown',e=>{if(e.key==='Enter')joinClass();});
  if($('prevBtn')) $('prevBtn').addEventListener('click',()=>teacherPost('/api/prev'));
  if($('nextBtn')) $('nextBtn').addEventListener('click',()=>teacherPost('/api/next'));
  if($('resetBtn')) $('resetBtn').addEventListener('click',()=>{if(confirm('Reset students, poll and lesson?')) teacherPost('/api/reset');});
  if($('pollBtn')) $('pollBtn').addEventListener('click',()=>{const q=prompt('Question for students:'); if(!q)return; const raw=prompt('Options separated by |','A|B|C|D'); if(!raw)return; teacherPost('/api/poll',{question:q,options:raw.split('|').map(x=>x.trim()).filter(Boolean)});});
  if($('joinView') && studentId){ $('joinView').hidden=true; $('classView').hidden=false; }
  if($('classView')) refreshStudent();
  if($('teacherContent')) refreshTeacher();
  setInterval(()=>{if($('classView')&&!$('classView').hidden)refreshStudent(); if($('teacherContent'))refreshTeacher();},1000);
}
document.addEventListener('DOMContentLoaded',setup);
