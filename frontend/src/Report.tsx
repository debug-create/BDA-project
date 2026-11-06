import type { ReactNode } from 'react'
import { Reveal } from './Reveal'

export function FigureBlock({number,kind='Figure',caption,source,meaning,children,chart=false}:{number:string;kind?:'Figure'|'Table';caption:string;source:string;meaning?:string;children:ReactNode;chart?:boolean}) {
  return <Reveal className="report-figure" drawOnce={chart}>
    <div className="figure-content">{children}</div>
    <p className="figure-caption"><strong>{kind} {number}.</strong> {caption}</p>
    <p className="source-line"><strong>Source:</strong> {source}</p>
    {meaning&&<details className="what-means"><summary>What this means</summary><p>{meaning}</p></details>}
  </Reveal>
}

export function KeyFindings({items}:{items:string[]}) {
  return <Reveal className="key-findings"><h3>Key findings</h3><ul>{items.map((item,index)=><li key={item} style={{'--row-index':index} as React.CSSProperties}>{item}</li>)}</ul></Reveal>
}

export function TechnicalNote({children}:{children:ReactNode}) {
  return <Reveal className="technical-note"><h3>Technical notes</h3><div>{children}</div></Reveal>
}
