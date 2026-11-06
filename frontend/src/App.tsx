import { useEffect, useMemo, useState } from 'react'
import { Activity, BarChart3, ChevronRight, ClipboardCheck, Database, Layers3, Moon, Plane, Server, Sun } from 'lucide-react'
import { copy, type RuntimeKey, type WindowKey } from './copy'
import { Reveal } from './Reveal'
import { FigureBlock, KeyFindings, TechnicalNote } from './Report'

type SiteData = any
type ModelData = any
type Page = 'overview'|'operations'|'holidays'|'models'|'method'|'data'

const pageIds: Page[] = ['overview','operations','holidays','models','method']
const pageIcons = [Plane,Server,BarChart3,Layers3,ClipboardCheck]
const routePage = () => {
  const value = window.location.hash.replace('#/','') as Page
  if([...pageIds,'data'].includes(value))return value
  window.location.hash='/overview'
  return 'overview'
}

const fmt = (value:number) => new Intl.NumberFormat('en-US').format(value)
const pct = (value:number, decimals:number) => `${(value * 100).toFixed(decimals)}%`

function Chip({children,tone='normal'}:{children:any;tone?:'normal'|'green'}) {
  return <span className={`chip chip-${tone}`}>{children}</span>
}
function SectionTitle({eyebrow,title,aside}:{eyebrow:string;title:string;aside?:string}) {
  return <div className="section-title"><div><span className="eyebrow">{eyebrow}</span><h2>{title}</h2></div>{aside&&<p>{aside}</p>}</div>
}
function Metric({label,value,detail,chip,primary=false}:{label:string;value:string;detail?:string;chip:string;primary?:boolean}) {
  return <article className={`stat ${primary?'primary-stat':''}`}><div className="stat-top"><span>{label}</span><Chip>{chip}</Chip></div><strong>{value}</strong>{detail&&<p>{detail}</p>}</article>
}

function SeasonStrip({data}:{data:SiteData}) {
  const [active,setActive]=useState<number|null>(null)
  const start = new Date('2024-10-01T00:00:00')
  const days = data.season.rates.map((rate:number,index:number) => {
    const date = new Date(start); date.setDate(start.getDate()+index)
    return {rate,label:date.toLocaleDateString('en-US',{month:'short',day:'numeric'})}
  })
  return <div className="season-wrap">
    <div className="season-head"><div><span className="eyebrow">{copy.season.eyebrow}</span><h3>{copy.season.title}</h3></div><div className="season-legend"><span>{copy.season.lower}</span><i/><span>{copy.season.higher}</span></div></div>
    <div className="holiday-band"><span className="halloween">{copy.season.halloween}</span><span className="thanks">{copy.season.thanksgiving}</span><span className="christmas">{copy.season.christmas}</span></div>
    <div className="season" role="list" aria-label={copy.season.aria}>{days.map((day:any,index:number)=><button key={day.label} onMouseEnter={()=>setActive(index)} onFocus={()=>setActive(index)} onMouseLeave={()=>setActive(null)} style={{'--heat':day.rate/Math.max(...data.season.rates)} as any} aria-label={`${day.label}: ${copy.season.ariaRate} ${day.rate}%`}><span/></button>)}</div>
    <div className="split-brackets"><span style={{flex:data.splits[0].flights}}>{copy.season.train}</span><span style={{flex:data.splits[1].flights}}>{copy.season.tuning}</span><span style={{flex:data.splits[2].flights}}>{copy.season.test}</span></div>
    <div className="season-foot"><span>{active===null?copy.season.idle:`${days[active].label} · ${copy.season.ariaRate} ${days[active].rate}%`}</span><span>{copy.season.source}</span></div>
  </div>
}

function HeadlineMetrics({run,modelData}:{run:any;modelData:ModelData}) {
  if(!run)return null
  const windowLabel=copy.global.windows[run.split as WindowKey]
  const ratio=run.auc_pr/run.prevalence
  return <div className="headline-metrics">
    <Metric primary label={copy.overview.primary} value={run.auc_roc.toFixed(modelData.display.auc_decimals)} detail={`${copy.overview.selectedModel} · ${run.model}`} chip={windowLabel}/>
    <div className="stats-grid headline-secondary">
      <Metric label={copy.overview.prVsBase} value={`${ratio.toFixed(modelData.display.percent_decimals)}${copy.overview.baseRatePhrase}`} chip={windowLabel}/>
      <Metric label={copy.overview.precision} value={pct(run.top_10_precision,modelData.display.percent_decimals)} chip={windowLabel}/>
      <Metric label={copy.overview.lift} value={`${run.top_10_lift.toFixed(modelData.display.lift_decimals)}${copy.overview.baseRatePhrase}`} chip={windowLabel}/>
    </div>
  </div>
}

function DidYouKnow({data,modelData}:{data:SiteData;modelData:ModelData}) {
  const [index,setIndex]=useState(0)
  const selected=modelData.runs.find((run:any)=>run.role==='selected'&&run.split==='test')
  const holiday=modelData.holiday_slices[0]
  const facts=[
    `${fmt(data.overview.eligibleRows)} ${copy.overview.tickerFacts.eligible}`,
    `${selected.auc_roc.toFixed(modelData.display.auc_decimals)} ${copy.overview.tickerFacts.auc}`,
    `${pct(selected.top_10_precision,modelData.display.percent_decimals)} ${copy.overview.tickerFacts.precision}`,
    `${pct(holiday.prevalence,modelData.display.percent_decimals)} ${copy.overview.tickerFacts.holiday}`,
  ]
  useEffect(()=>{
    const timer=window.setInterval(()=>{if(!document.hidden)setIndex(current=>(current+1)%facts.length)},5000)
    return()=>window.clearInterval(timer)
  },[facts.length])
  return <Reveal className="fact-ticker"><strong>{copy.overview.tickerTitle}</strong><span key={index}>{facts[index]}</span></Reveal>
}

function Overview({data,modelData,setPage}:{data:SiteData;modelData:ModelData;setPage:(p:Page)=>void}) {
  const selected=modelData.runs.find((run:any)=>run.role==='selected'&&run.split==='test')
  return <>
    <Reveal><section className="hero compact-hero"><div className="hero-copy"><span className="eyebrow">{copy.overview.eyebrow}</span><h1>{copy.overview.titleLineOne}<br/><em>{copy.overview.titleLineTwo}</em></h1><p>{copy.overview.question}</p>{selected&&<p>{copy.overview.resultPrefix} {selected.top_10_lift.toFixed(modelData.display.lift_decimals)}× {copy.overview.resultSuffix}</p>}<p>{copy.overview.stack}</p></div><div className="airport-code"><Plane/><strong>OCT</strong><span>→</span><strong>DEC</strong><small>{copy.overview.airport}</small></div></section></Reveal>
    <FigureBlock number="1" caption={copy.models.figureHeadline} source={copy.global.sourceProject} meaning={copy.models.meaningHeadline}><HeadlineMetrics run={selected} modelData={modelData}/></FigureBlock>
    <DidYouKnow data={data} modelData={modelData}/>
    <FigureBlock number="2" caption={copy.overview.figureSeason} source={copy.global.sourceBts} meaning={copy.overview.meaningSeason} chart><SeasonStrip data={data}/></FigureBlock>
    <section><SectionTitle eyebrow={copy.overview.pipelineEyebrow} title={copy.overview.pipelineTitle}/><FigureBlock number="3" caption={copy.overview.figurePipeline} source={copy.global.sourceProject} meaning={copy.overview.meaningPipeline}><div className="route-line">{data.pipeline.map((step:any,index:number)=><button key={step.name} onClick={()=>setPage('operations')}><b>{String(index+1).padStart(data.display.step_digits,'0')}</b><span>{step.name}</span>{index<data.pipeline.length-1&&<ChevronRight/>}</button>)}</div></FigureBlock></section>
    <Reveal><section><SectionTitle eyebrow={copy.overview.linksTitle} title={copy.overview.question}/><div className="overview-links"><button onClick={()=>setPage('holidays')}>{copy.overview.holidaysLink}<ChevronRight/></button><button onClick={()=>setPage('models')}>{copy.overview.modelsLink}<ChevronRight/></button></div></section></Reveal>
  </>
}

function DataPage({data}:{data:SiteData}) {
  return <><SectionTitle eyebrow={copy.data.eyebrow} title={copy.data.title} aside={copy.data.aside}/>
    <div className="provenance"><div><span>{copy.data.datasetWindow}</span><strong>{data.meta.window}</strong></div><div><span>{copy.data.rawRows}</span><strong>{fmt(data.overview.rawRows)}</strong></div><Chip>{copy.data.clusterRun}</Chip></div>
    <section><SectionTitle eyebrow={copy.data.eligibilityEyebrow} title={copy.data.eligibilityTitle}/><FigureBlock number="A1" caption={copy.data.figureEligibility} source={copy.global.sourceBts}><div className="funnel">{data.eligibility.map((row:any,index:number)=><div key={row.label} style={{width:`${row.value/data.eligibility[0].value*100}%`,'--row-index':index} as React.CSSProperties}><span>{row.label}</span><strong>{fmt(row.value)}</strong><small>{row.note}</small></div>)}</div></FigureBlock></section>
    <section><SectionTitle eyebrow={copy.data.splitEyebrow} title={copy.data.splitTitle}/><FigureBlock number="A1" kind="Table" caption={copy.data.tableWindows} source={copy.global.sourceBts}><div className="split-cards">{data.splits.map((split:any,index:number)=><article key={split.name} style={{'--row-index':index} as React.CSSProperties}><span className="eyebrow">{copy.global.windows[index===0?'train':index===1?'validation':'test']}</span><strong>{split.start} → {split.end}</strong><b>{fmt(split.flights)}</b><p>{pct(split.prevalence,data.display.percent_decimals)} {copy.data.delayed}</p><div className="bar"><i style={{width:pct(split.prevalence,data.display.percent_decimals)}}/></div></article>)}</div></FigureBlock></section>
  </>
}

function Operations({data}:{data:SiteData}) {
  return <><SectionTitle eyebrow={copy.operations.eyebrow} title={copy.operations.title} aside={copy.operations.aside}/>
    <FigureBlock number="4" caption={copy.operations.figurePipeline} source={copy.global.sourceProject}><div className="route-map">{data.pipeline.map((step:any,index:number)=><div className="route-static" key={step.name}><span>{String(index+1).padStart(data.display.step_digits,'0')}</span><strong>{step.name}</strong><small>{step.detail}</small><code>{step.path}</code></div>)}</div></FigureBlock>
    <section><SectionTitle eyebrow={copy.operations.topologyEyebrow} title={copy.operations.topologyTitle}/><FigureBlock number="5" caption={copy.operations.figureTopology} source={copy.global.sourceProject}><div className="cluster"><div className="master"><Server/><strong>{data.cluster[0].name}</strong><span>{data.cluster[0].roles}</span></div><div className="workers">{data.cluster.slice(1).map((node:any)=><div key={node.name}><Activity/><strong>{node.name}</strong><span>{node.roles}</span><Chip tone="green">{node.state}</Chip></div>)}</div></div></FigureBlock></section>
    <section className="config"><SectionTitle eyebrow={copy.operations.snapshotEyebrow} title={copy.operations.snapshotTitle} aside={`${copy.operations.recordedOn} ${data.clusterSnapshot.captured_on}`}/><FigureBlock number="1" kind="Table" caption={copy.operations.tableSettings} source={copy.global.sourceProject}><div className="config-grid">{data.clusterSnapshot.spark_settings.map((setting:any,index:number)=><div key={setting.label} style={{'--row-index':index} as React.CSSProperties}><span>{setting.label}</span><code>{setting.value}</code></div>)}</div></FigureBlock></section>
    <section><SectionTitle eyebrow={copy.operations.hdfsEyebrow} title={copy.operations.hdfsTitle}/><FigureBlock number="2" kind="Table" caption={copy.operations.tablePaths} source={copy.global.sourceProject}><div className="commands">{data.clusterSnapshot.hdfs_paths.map((path:string)=><code key={path}>{path}</code>)}</div></FigureBlock></section>
  </>
}

function DailyLine({rates}:{rates:number[]}) {
  if(!rates?.length)return null
  const [first,setFirst]=useState(0), [second,setSecond]=useState(rates.length-1)
  const max=Math.max(...rates), min=Math.min(...rates)
  const x=(index:number)=>index/(rates.length-1)*100
  const y=(value:number)=>40-(value-min)/(max-min)*34
  const points=rates.map((value,index)=>`${x(index)},${y(value)}`).join(' ')
  const rangeStart=Math.min(x(first),x(second)), rangeWidth=Math.abs(x(second)-x(first))
  const label=(index:number)=>{const date=new Date('2024-10-01T00:00:00');date.setDate(date.getDate()+index);return date.toLocaleDateString('en-US',{month:'short',day:'numeric'})}
  const middle=Math.floor((rates.length-1)/2), midRate=Math.round((max+min)/2)
  return <><div className="daily-chart-frame"><strong className="y-axis-title">{copy.holidays.delayRateAxis}</strong><div className="y-axis-ticks"><span>{max}%</span><span>{midRate}%</span><span>{min}%</span></div><svg className="daily-line" viewBox="0 0 100 40" preserveAspectRatio="none" aria-label={copy.holidays.dailyTitle}><rect className="date-selection" x={rangeStart} y="0" width={rangeWidth} height="40"/><polyline pathLength="1" points={points}/><line className="date-marker first" x1={x(first)} x2={x(first)} y1="0" y2="40"/><line className="date-marker second" x1={x(second)} x2={x(second)} y1="0" y2="40"/><circle className="date-point first" cx={x(first)} cy={y(rates[first])} r="1.1"/><circle className="date-point second" cx={x(second)} cy={y(rates[second])} r="1.1"/></svg><div className="x-axis-ticks"><span>{label(0)}</span><span>{label(middle)}</span><span>{label(rates.length-1)}</span></div><strong className="x-axis-title">{copy.holidays.dateAxis}</strong></div><div className="date-brush"><strong>{copy.holidays.brushTitle}</strong><label>{copy.holidays.firstDate}<input aria-label={copy.holidays.firstDate} type="range" min={0} max={rates.length-1} value={first} onChange={event=>setFirst(Number(event.target.value))}/><span>{label(first)} · {rates[first]}%</span></label><label>{copy.holidays.secondDate}<input aria-label={copy.holidays.secondDate} type="range" min={0} max={rates.length-1} value={second} onChange={event=>setSecond(Number(event.target.value))}/><span>{label(second)} · {rates[second]}%</span></label><p aria-live="polite">{copy.holidays.difference}: {Math.abs(rates[second]-rates[first])} pp</p></div></>
}

function Holidays({data,modelData}:{data:SiteData;modelData:ModelData}) {
  const [tab,setTab]=useState<'airline'|'hour'>('hour')
  const hasAirlines=data.holidayBreakdowns.airline?.length>0
  const findingRows=[`${modelData.holiday_slices[0].name}: ${pct(modelData.holiday_slices[0].prevalence,modelData.display.percent_decimals)} ${copy.holidays.keyFindings[0]}`,copy.holidays.keyFindings[1],copy.holidays.keyFindings[2]]
  return <><SectionTitle eyebrow={copy.holidays.eyebrow} title={copy.holidays.title} aside={copy.holidays.aside}/>
    <KeyFindings items={findingRows}/>
    <section><SectionTitle eyebrow={copy.holidays.comparisonEyebrow} title={copy.holidays.comparisonTitle}/><FigureBlock number="3" kind="Table" caption={copy.holidays.tableComparison} source={copy.global.sourceBts} meaning={copy.holidays.meaningComparison}><div className="evidence-table"><table><thead><tr><th>{copy.holidays.slice}</th><th>{copy.holidays.flights}</th><th>{copy.holidays.baseRate}</th><th>{copy.holidays.metric}</th></tr></thead><tbody>{modelData.holiday_slices.map((row:any,index:number)=><tr key={row.name} style={{'--row-index':index} as React.CSSProperties}><td>{row.name}</td><td>{fmt(row.flights)}</td><td>{pct(row.prevalence,modelData.display.percent_decimals)}</td><td><strong>{row.auc_pr.toFixed(modelData.display.metric_decimals)}</strong></td></tr>)}</tbody></table></div></FigureBlock></section>
    <section><SectionTitle eyebrow={copy.holidays.dailyEyebrow} title={copy.holidays.dailyTitle}/><FigureBlock number="6" caption={copy.holidays.figureDaily} source={copy.global.sourceBts} meaning={copy.holidays.meaningDaily} chart><div className="line-panel"><DailyLine rates={data.season.rates}/><div className="line-bands"><span className="halloween">{copy.season.halloween}</span><span className="thanks">{copy.season.thanksgiving}</span><span className="christmas">{copy.season.christmas}</span></div></div></FigureBlock></section>
    <section><SectionTitle eyebrow={copy.holidays.breakdownEyebrow} title={copy.holidays.breakdownTitle}/><div className="segmented">{hasAirlines&&<button className={tab==='airline'?'active':''} onClick={()=>setTab('airline')}>{copy.holidays.airlineTab}</button>}<button className={tab==='hour'?'active':''} onClick={()=>setTab('hour')}>{copy.holidays.hourTab}</button></div>{tab==='hour'&&data.holidayBreakdowns.departureHour?.length>0&&<FigureBlock number="7" caption={copy.holidays.figureHour} source={copy.global.sourceBts} chart><div className="hour-chart" aria-label={copy.holidays.hourChart}>{data.holidayBreakdowns.departureHour.map((value:number,index:number)=><div key={index}><i style={{height:`${value/Math.max(...data.holidayBreakdowns.departureHour)*100}%`}}/><span>{index}</span></div>)}</div></FigureBlock>}</section>
    <TechnicalNote>{copy.holidays.technical}</TechnicalNote>
    <Reveal><section><SectionTitle eyebrow={copy.holidays.descriptive} title={copy.holidays.causesTitle}/><p className="section-copy">{copy.holidays.causesBody}</p></section></Reveal>
  </>
}

function RunCard({run,modelData,large=false}:{run:any;modelData:ModelData;large?:boolean}) {
  if(!run)return null
  return <article className={`boarding-pass ${large?'selected model-large':''}`}><div className="pass-head"><Chip tone={large?'green':'normal'}>{large?copy.models.selected:copy.models.reference}</Chip><Chip>{copy.models.runsOn} · {copy.global.runtime[run.executed_on as RuntimeKey]}</Chip></div><h3>{run.model}</h3><div className="pass-route"><div><span>{copy.models.window}</span><b>{copy.global.windows[run.split as WindowKey]}</b></div><Plane/><div><span>{copy.models.auc}</span><b>{run.auc_roc.toFixed(modelData.display.auc_decimals)}</b></div></div><div className="pass-metrics"><div><span>{copy.models.prauc}</span><strong>{run.auc_pr.toFixed(modelData.display.metric_decimals)}</strong></div><div><span>{copy.models.baseRate}</span><strong>{pct(run.prevalence,modelData.display.percent_decimals)}</strong></div><div><span>{copy.models.topPrecision}</span><strong>{pct(run.top_10_precision,modelData.display.percent_decimals)}</strong></div><div><span>{copy.models.lift}</span><strong>{run.top_10_lift.toFixed(modelData.display.lift_decimals)}{copy.models.baseRatePhrase}</strong></div></div></article>
}

function CurveChart({points,title,xLabel,yLabel,base}:{points:any[];title:string;xLabel:string;yLabel:string;base?:number}) {
  if(!points?.length)return null
  const coords=points.map((point:any)=>`${point[0]*100},${100-point[1]*100}`).join(' ')
  return <article className="curve-card"><h3>{title}</h3><svg viewBox="0 0 100 100" preserveAspectRatio="none">{base!==undefined&&<line x1="0" x2="100" y1={100-base*100} y2={100-base*100}/>}<polyline pathLength="1" points={coords}/></svg><div><span>{yLabel}</span><span>{xLabel}</span></div></article>
}

function Models({modelData}:{modelData:ModelData}) {
  const selected=modelData.runs.find((run:any)=>run.role==='selected'&&run.split==='test')
  const reference=modelData.runs.find((run:any)=>run.role==='reference'&&run.split==='test')
  const [tab,setTab]=useState<'risk'|'threshold'>('risk')
  const [riskIndex,setRiskIndex]=useState(0), [thresholdIndex,setThresholdIndex]=useState(0)
  const risk=modelData.ranking[riskIndex], threshold=modelData.thresholds[thresholdIndex]
  const findings=[`${selected.model}: ${selected.auc_roc.toFixed(modelData.display.auc_decimals)} ${copy.models.keyFindings[0]}`,`${selected.top_10_lift.toFixed(modelData.display.lift_decimals)}${copy.models.baseRatePhrase}. ${copy.models.keyFindings[1]}`,copy.models.keyFindings[2]]
  return <><SectionTitle eyebrow={copy.models.eyebrow} title={copy.models.title} aside={copy.models.aside}/>
    <KeyFindings items={findings}/>
    <FigureBlock number="8" caption={copy.models.figureHeadline} source={copy.global.sourceProject} meaning={copy.models.meaningHeadline}><HeadlineMetrics run={selected} modelData={modelData}/></FigureBlock>
    <FigureBlock number="4" kind="Table" caption={copy.models.tableModels} source={copy.global.sourceProject}><div className="selected-models"><RunCard run={selected} modelData={modelData} large/><RunCard run={reference} modelData={modelData}/></div></FigureBlock>
    {selected?.curves&&(selected.curves.pr?.length||selected.curves.roc?.length)&&<section><SectionTitle eyebrow={copy.models.curvesEyebrow} title={`${copy.models.prCurve} · ${copy.models.rocCurve}`}/><div className="two-col"><CurveChart points={selected.curves.pr} title={copy.models.prCurve} xLabel={copy.models.recall} yLabel={copy.models.precision} base={selected.prevalence}/><CurveChart points={selected.curves.roc} title={copy.models.rocCurve} xLabel={copy.models.falsePositive} yLabel={copy.models.truePositive}/></div></section>}
    <section><FigureBlock number="9" caption={copy.models.figureExplorer} source={copy.global.sourceProject} meaning={copy.models.meaningExplorer}><div className="segmented"><button className={tab==='risk'?'active':''} onClick={()=>setTab('risk')}>{copy.models.tabs[0]}</button><button className={tab==='threshold'?'active':''} onClick={()=>setTab('threshold')}>{copy.models.tabs[1]}</button></div>{tab==='risk'&&risk&&<div className="risk"><input aria-label={copy.models.capacity} type="range" min={0} max={modelData.ranking.length-1} step={1} value={riskIndex} onChange={event=>setRiskIndex(Number(event.target.value))} disabled={modelData.ranking.length===1}/><div className="risk-board"><span>{copy.models.capacity}</span><strong>{pct(risk.target_fraction,modelData.display.percent_decimals)}</strong><span>{copy.models.riskPrecision}</span><strong>{pct(risk.precision,modelData.display.percent_decimals)}</strong><span>{copy.models.riskRecall}</span><strong>{pct(risk.recall,modelData.display.percent_decimals)}</strong><span>{copy.models.liftVsBase}</span><strong>{risk.lift.toFixed(modelData.display.lift_decimals)}{copy.models.baseRatePhrase}</strong></div></div>}{tab==='threshold'&&threshold&&<div className="risk"><input aria-label={copy.models.threshold} type="range" min={0} max={modelData.thresholds.length-1} step={1} value={thresholdIndex} onChange={event=>setThresholdIndex(Number(event.target.value))}/><div className="risk-board"><span>{copy.models.threshold}</span><strong>{threshold.threshold}</strong><span>{copy.models.riskPrecision}</span><strong>{pct(threshold.precision,modelData.display.percent_decimals)}</strong><span>{copy.models.riskRecall}</span><strong>{pct(threshold.recall,modelData.display.percent_decimals)}</strong><span>{copy.models.flagged}</span><strong>{pct(threshold.flagged,modelData.display.percent_decimals)}</strong></div></div>}</FigureBlock></section>
    {selected?.confusion&&<section><SectionTitle eyebrow={copy.models.confusionEyebrow} title={copy.models.confusionTitle}/><div className="confusion">{Object.entries(selected.confusion).map(([key,value])=><div key={key}><span>{key}</span><strong>{fmt(value as number)}</strong></div>)}</div></section>}
    {modelData.feature_importance?.length>0&&<section><SectionTitle eyebrow={copy.models.importanceEyebrow} title={copy.models.importanceTitle}/><FigureBlock number="10" caption={copy.models.figureImportance} source={copy.global.sourceProject} meaning={copy.models.meaningImportance} chart><div className="importance">{modelData.feature_importance.slice(0,10).map((row:any,index:number)=><div key={row.name} style={{'--row-index':index} as React.CSSProperties}><span>{row.name}</span><i style={{width:pct(row.value/modelData.feature_importance[0].value,modelData.display.percent_decimals)}}/><b>{row.value.toFixed(modelData.display.metric_decimals)}</b></div>)}</div><p className="fine">{copy.models.importanceNote}</p></FigureBlock></section>}
    <section><SectionTitle eyebrow={copy.models.slicesEyebrow} title={copy.models.slicesTitle}/><FigureBlock number="5" kind="Table" caption={copy.models.tableSlices} source={copy.global.sourceProject}><div className="evidence-table"><table><thead><tr><th>{copy.holidays.slice}</th><th>{copy.holidays.baseRate}</th><th>{copy.models.prauc}</th></tr></thead><tbody>{modelData.holiday_slices.map((row:any,index:number)=><tr key={row.name} style={{'--row-index':index} as React.CSSProperties}><td>{row.name}</td><td>{pct(row.prevalence,modelData.display.percent_decimals)}</td><td><strong>{row.auc_pr.toFixed(modelData.display.metric_decimals)}</strong></td></tr>)}</tbody></table></div></FigureBlock></section>
  </>
}

function Method({data}:{data:SiteData}) {
  return <><SectionTitle eyebrow={copy.method.eyebrow} title={copy.method.title} aside={copy.method.aside}/><FigureBlock number="11" caption={copy.method.figureInputs} source={copy.global.sourceProject}><section className="method-inputs"><h3>{copy.method.inputsTitle}</h3>{copy.method.inputs.map((item,index)=><p key={item} style={{'--row-index':index} as React.CSSProperties}><span>✓</span>{item}</p>)}</section></FigureBlock><section><SectionTitle eyebrow={copy.method.timelineEyebrow} title={copy.method.timelineTitle}/><FigureBlock number="12" caption={copy.method.figureTimeline} source={copy.global.sourceProject}><div className="timeline">{data.splits.map((split:any,index:number)=><div key={split.name} style={{'--row-index':index} as React.CSSProperties}><span className="eyebrow">{copy.global.windows[index===0?'train':index===1?'validation':'test']}</span><strong>{split.start} — {split.end}</strong><p>{copy.method.timelineNotes[index]}</p></div>)}</div></FigureBlock></section><TechnicalNote>{copy.method.technical}</TechnicalNote></>
}

function App(){
  const [data,setData]=useState<SiteData|null>(null), [modelData,setModelData]=useState<ModelData|null>(null)
  const [page,setPageState]=useState<Page>(routePage), [theme,setTheme]=useState('dark')
  useEffect(()=>{Promise.all([fetch('./data/site-data.json').then(r=>r.json()),fetch('./data/model_runs.json').then(r=>r.json())]).then(([site,models])=>{setData(site);setModelData(models)})},[])
  useEffect(()=>{const onHash=()=>setPageState(routePage());window.addEventListener('hashchange',onHash);return()=>window.removeEventListener('hashchange',onHash)},[])
  useEffect(()=>{document.documentElement.dataset.theme=theme;localStorage.setItem('theme',theme)},[theme])
  useEffect(()=>{const sync=()=>document.documentElement.dataset.pageHidden=String(document.hidden);sync();document.addEventListener('visibilitychange',sync);return()=>document.removeEventListener('visibilitychange',sync)},[])
  useEffect(()=>{window.scrollTo({top:0,behavior:'smooth'})},[page])
  const setPage=(next:Page)=>{window.location.hash=`/${next}`;setPageState(next)}
  const pages=useMemo(()=>pageIds.map((id,index)=>({id,label:copy.global.pages[id][0],board:copy.global.pages[id][1],icon:pageIcons[index]})),[])
  if(!data||!modelData)return <div className="loading"><Plane/><span>{copy.global.loading}</span></div>
  const views:Record<Page,any>={overview:<Overview data={data} modelData={modelData} setPage={setPage}/>,operations:<Operations data={data}/>,holidays:<Holidays data={data} modelData={modelData}/>,models:<Models modelData={modelData}/>,method:<Method data={data}/>,data:<DataPage data={data}/>}
  const current=page==='data'?{label:copy.global.pages.data[0],board:copy.global.pages.data[1]}:pages.find(item=>item.id===page)!
  const sectionNumber=page==='data'?'A':String(pageIds.indexOf(page)+1)
  return <div className="app"><header className="status theme-only-header"><button className="theme" onClick={()=>setTheme(theme==='dark'?'light':'dark')} aria-label={copy.global.toggleTheme}>{theme==='dark'?<Sun/>:<Moon/>}</button></header><aside className="sidebar"><div className="brand"><div className="brand-mark"><Plane/></div><div><strong>{copy.global.brandTop}</strong><span>{copy.global.brandBottom}</span></div></div><nav>{pages.map((item,index)=>{const Icon=item.icon;return <button className={page===item.id?'active':''} key={item.id} onClick={()=>setPage(item.id)}><Icon/><span>{index+1}. {item.label}<small>{item.board}</small></span></button>})}</nav><div className="sidebar-foot"><span>{copy.global.siteType}</span><b>{copy.global.siteState}</b></div></aside><main><div className="breadcrumb"><button onClick={()=>setPage('overview')}>{copy.global.breadcrumbHome}</button><span>/</span><span>{current.label}</span></div><Reveal><div className="page-head"><div><span className="eyebrow">{current.board}</span><h4>{sectionNumber}. {current.label}</h4></div></div></Reveal>{views[page]}<footer><div><Plane/><strong>{copy.global.appTitle}</strong></div><p>{copy.global.footerAttribution}</p></footer></main></div>
}

export default App
