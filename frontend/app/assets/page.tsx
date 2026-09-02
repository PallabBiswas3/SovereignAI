"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

const api = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Asset = { asset_id:string; canonical_name:string; asset_type:string; plant_id:string; area_id:string; unit_id:string; criticality:string; status:string; manufacturer?:string|null; model?:string|null; aliases:string[] };
type Measurement = { measurement_id:string; metric:string; value:number; unit:string; timestamp:string; quality:string; freshness_status:string; age_seconds?:number|null; warnings:string[]; original_value:number; original_unit:string };
type Trend = { latest:number; mean:number; minimum:number; maximum:number; change:number; slope_per_day:number; trend:string; sample_count:number };
type TimelineItem = { id:string; timestamp:string; type:string; title:string };
type Maintenance = { id:string; occurred_at:string; event_type:string; title:string; summary:string };

const label = (value:string) => value.replaceAll("_", " ").replace(/\b\w/g, (part) => part.toUpperCase());
function age(seconds?:number|null) { if (seconds == null) return "Unknown age"; if (seconds < 60) return `${Math.round(seconds)} sec ago`; if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`; return `${(seconds / 3600).toFixed(1)} hr ago`; }

export default function AssetsPage() {
  const [assets,setAssets]=useState<Asset[]>([]); const [selected,setSelected]=useState("Pump-102");
  const [asset,setAsset]=useState<Asset|null>(null); const [measurements,setMeasurements]=useState<Measurement[]>([]);
  const [trend,setTrend]=useState<Trend|null>(null); const [series,setSeries]=useState<Measurement[]>([]);
  const [timeline,setTimeline]=useState<TimelineItem[]>([]); const [maintenance,setMaintenance]=useState<Maintenance[]>([]);
  const [error,setError]=useState("");

  useEffect(()=>{ fetch(`${api}/api/assets`,{credentials:"include"}).then(async response=>{if(!response.ok)throw new Error("Sign in from the workbench to view authorized assets.");return response.json();}).then(payload=>setAssets(payload.items??[])).catch(caught=>setError(caught instanceof Error?caught.message:"Assets unavailable")); },[]);
  useEffect(()=>{ setError(""); Promise.all([
    fetch(`${api}/api/assets/${encodeURIComponent(selected)}`,{credentials:"include"}),
    fetch(`${api}/api/assets/${encodeURIComponent(selected)}/measurements/latest`,{credentials:"include"}),
    fetch(`${api}/api/assets/${encodeURIComponent(selected)}/timeline`,{credentials:"include"}),
    fetch(`${api}/api/assets/${encodeURIComponent(selected)}/maintenance`,{credentials:"include"}),
    fetch(`${api}/api/assets/${encodeURIComponent(selected)}/measurements/history?metric=vibration`,{credentials:"include"}),
  ]).then(async responses=>{if(!responses[0].ok)throw new Error(`Asset access failed (${responses[0].status}).`); const [passport,latest,time,work,history]=await Promise.all(responses.map(response=>response.ok?response.json():Promise.resolve({}))); setAsset(passport.asset??null);setMeasurements(latest.measurements??[]);setTimeline(time.items??[]);setMaintenance(work.history??[]);setTrend(history.trend??null);setSeries(history.series?.measurements??[]);}).catch(caught=>setError(caught instanceof Error?caught.message:"Asset view unavailable")); },[selected]);

  const vibration=measurements.find(item=>item.metric==="vibration"); const maximum=Math.max(...series.map(item=>item.value),1);
  return <main className="assetPage">
    <header className="monitorHeader"><div><small>ASSET INTELLIGENCE · READ ONLY</small><h1>Plant asset context</h1></div><Link href="/">Back to workbench</Link></header>
    <div className="simulatedBanner"><b>SIMULATED PLANT DATA</b><span>Deterministic APEL demonstration — no live refinery connection and no plant-control commands.</span></div>
    <div className="assetToolbar"><label>AUTHORIZED ASSET<select value={selected} onChange={event=>setSelected(event.target.value)}>{assets.map(item=><option value={item.asset_id} key={item.asset_id}>{item.asset_id} · {item.canonical_name}</option>)}</select></label><span className="readonlyPill">READ ONLY</span></div>
    {error&&<p className="error">{error}</p>}
    {asset&&<><section className="assetHero"><div><small>{asset.asset_id}</small><h2>{asset.canonical_name}</h2><p>{label(asset.asset_type)} · {asset.plant_id} / {asset.area_id} / {asset.unit_id}</p></div><div className="assetStatus"><span>{label(asset.status)}</span><b>{asset.criticality} CRITICALITY</b></div></section>
    <div className="assetGrid">
      <section className="assetCard passportCard"><label>ASSET PASSPORT</label><div className="assetFacts"><span>Asset ID<b>{asset.asset_id}</b></span><span>Type<b>{label(asset.asset_type)}</b></span><span>Plant<b>{asset.plant_id}</b></span><span>Area<b>{asset.area_id}</b></span><span>Unit<b>{asset.unit_id}</b></span><span>Manufacturer / model<b>{asset.manufacturer??"—"} {asset.model??""}</b></span></div><p>Aliases: {asset.aliases.join(", ")||"None"}</p></section>
      <section className="assetCard"><label>LATEST AUTHORIZED READINGS</label>{measurements.length?measurements.map(item=><article className="reading" key={item.measurement_id}><div><b>{label(item.metric)}</b><small>{new Date(item.timestamp).toLocaleString()}</small></div><strong>{item.original_value} {item.original_unit}</strong><div><span className={item.quality==="GOOD"?"qualityGood":"qualityWarn"}>{item.quality}</span><small>{item.freshness_status} · {age(item.age_seconds)}</small></div></article>):<p>No authorized readings available.</p>}</section>
      <section className="assetCard trendCard"><label>VIBRATION HISTORY · DETERMINISTIC</label>{trend?<><div className="trendSummary"><strong>{trend.latest} mm/s</strong><span className={trend.trend==="INCREASING"?"qualityWarn":"qualityGood"}>{trend.trend}</span></div><div className="trendBars" aria-label="Historical vibration readings">{series.map(item=><div key={item.measurement_id}><i style={{height:`${Math.max(8,item.value/maximum*100)}%`}} title={`${item.value} ${item.unit}`}/><small>{new Date(item.timestamp).toLocaleDateString(undefined,{month:"short"})}</small></div>)}</div><div className="assetFacts"><span>Mean<b>{trend.mean.toFixed(2)}</b></span><span>Range<b>{trend.minimum}–{trend.maximum}</b></span><span>Change<b>{trend.change>0?"+":""}{trend.change}</b></span><span>Samples<b>{trend.sample_count}</b></span></div><p>This is condition and threshold analysis, not failure prediction.</p></>:<p>No sufficient vibration history.</p>}</section>
      <section className="assetCard"><label>ASSET TIMELINE</label><div className="assetTimeline">{timeline.map(item=><article key={item.id}><i/><div><small>{new Date(item.timestamp).toLocaleDateString()}</small><b>{item.type}</b><p>{item.title}</p></div></article>)}</div></section>
      <section className="assetCard"><label>MAINTENANCE HISTORY</label>{maintenance.map(item=><details key={item.id}><summary>{item.title}</summary><small>{new Date(item.occurred_at).toLocaleDateString()} · {item.event_type}</small><p>{item.summary}</p></details>)}{!maintenance.length&&<p>No authorized maintenance records.</p>}</section>
      <section className="assetCard whyCard"><label>WHY THIS CONDITION?</label>{vibration?<><p><b>Asset:</b> {asset.asset_id}</p><p><b>Current measurement:</b> {vibration.original_value} {vibration.original_unit} · {vibration.quality} · {vibration.freshness_status}</p><p><b>Trend:</b> {trend?.trend??"Insufficient history"} across {trend?.sample_count??0} readings</p><p><b>Source:</b> {vibration.measurement_id} · {vibration.timestamp}</p></>:<p>No supported measurement is available.</p>}</section>
    </div></>}
  </main>;
}
