// 最小DOM上で実際のイベントハンドラを呼ぶ。ブラウザのレイアウトは検証しない。
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { webcrypto } from 'node:crypto';
const source = readFileSync(new URL('../src/command_menu.js', import.meta.url), 'utf8');
const {default: mount} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
class Element {
  constructor(tag='div') { this.tag=tag; this.children=[]; this.style={}; this.attrs={}; this.listeners={}; this.value=''; this.disabled=false; this.classList=[]; }
  appendChild(c) { this.children.push(c); return c; }
  append(...items) { items.forEach(c=>this.appendChild(c)); }
  replaceChildren(...items) { this.children=items; }
  setAttribute(k,v) {this.attrs[k]=v;}
  getAttribute(k) {return this.attrs[k]??null;}
  removeAttribute(k) {delete this.attrs[k];}
  addEventListener(k,f) {this.listeners[k]=f;}
  removeEventListener(k) {delete this.listeners[k];}
  remove() {}
  scrollIntoView() {}
  contains(target) {return this===target||this.children.some(c=>c.contains(target));}
  closest(selector) {
    if (selector.includes('stChatInput')) return this;
    if (selector.includes('input')||selector.includes('contenteditable')) return ['textarea','input'].includes(this.tag)?this:null;
    return selector==='button' && this.tag==='button' ? this:null;
  }
  getBoundingClientRect() {return {left:10,top:500,width:600,height:40};}
  querySelectorAll() {return this.children.filter(c=>c.attrs.role==='option');}
  focus() {document.activeElement=this; this.listeners.focus?.();}
}
globalThis.Element=Element;
if (!globalThis.crypto) Object.defineProperty(globalThis, 'crypto', {value:webcrypto});
globalThis.MutationObserver=class {observe(){} disconnect(){}};
globalThis.getComputedStyle=()=>({getPropertyValue:()=>'',color:'black',backgroundColor:'white'});
globalThis.window={innerHeight:800,addEventListener(){},removeEventListener(){}};
function setup(effortOpen=true) {
  const input=new Element('textarea');
  const doc={head:new Element(),body:new Element(),activeElement:input,listeners:{},
    createElement:tag=>new Element(tag),
    querySelector:selector=>selector==='.st-key-prompt_input textarea'?input:null,
    addEventListener(k,f){this.listeners[k]=f;}, removeEventListener(k){delete this.listeners[k];}};
  globalThis.document=doc;
  const events=[];
  const cleanup=mount({data:{effort_open:effortOpen,effort:'medium',commands:[]},parentElement:new Element(),setTriggerValue:(key,value)=>events.push(value)});
  const menu=doc.body.children[0];
  function press(key, extra={}) {
    const event={key,target:input,isComposing:false,preventDefault(){this.prevented=true;},stopImmediatePropagation(){this.stopped=true;},...extra};
    doc.listeners.keydown(event);
    return event;
  }
  return {input,menu,press,events,cleanup};
}
for (const [key,mode] of [['1','low'],['2','medium'],['3','high']]) {
  const ui=setup();
  assert.equal(ui.menu.children[Number(key)-1].children[0].textContent.startsWith(key+'. '),true);
  const event=ui.press(key);
  assert.deepEqual(ui.events,[{kind:'effort',mode}]);
  assert.equal(event.prevented,true);
  assert.equal(ui.input.value,'');
  assert.equal(ui.menu.hidden,true);
  ui.cleanup();
}
for (const extra of [{isComposing:true},{keyCode:229},{ctrlKey:true},{shiftKey:true},{metaKey:true},{altKey:true},{target:new Element('input')}]) {
  const ui=setup(); ui.press('3',extra); assert.equal(ui.events.length,0); ui.cleanup();
}
const closed=setup(false); closed.press('1'); assert.equal(closed.events.length,0); closed.cleanup();
const arrows=setup(); arrows.press('ArrowDown'); arrows.press('Enter'); assert.deepEqual(arrows.events,[{kind:'effort',mode:'high'}]); arrows.cleanup();
const escaped=setup(); escaped.press('Escape'); assert.deepEqual(escaped.events,[{kind:'close_effort'}]); escaped.cleanup();
console.log('effort keyboard checks passed');
