import type { CSSProperties, ReactNode } from 'react'
import { useReveal } from './useReveal'

export function Reveal({children,className='',delay=0,drawOnce=false}:{children:ReactNode;className?:string;delay?:number;drawOnce?:boolean}) {
  const ref = useReveal<HTMLDivElement>(drawOnce)
  return <div ref={ref} className={`reveal ${drawOnce?'chart-reveal':''} ${className}`} style={{'--reveal-delay':`${delay}ms`} as CSSProperties}>{children}</div>
}
