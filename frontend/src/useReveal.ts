import { useEffect, useRef } from 'react'

export function useReveal<T extends HTMLElement>(drawOnce = false) {
  const ref = useRef<T>(null)

  useEffect(() => {
    const element = ref.current
    if (!element) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    element.dataset.revealReady = 'true'
    if (reduced || !('IntersectionObserver' in window)) {
      element.dataset.revealState = 'visible'
      return
    }

    element.dataset.revealState = 'below'
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        element.dataset.revealState = 'visible'
        if (drawOnce) observer.unobserve(element)
      } else {
        element.dataset.revealState = entry.boundingClientRect.top < 0 ? 'past' : 'below'
      }
    }, { threshold: 0.12, rootMargin: '0px 0px -6% 0px' })
    observer.observe(element)
    return () => observer.disconnect()
  }, [drawOnce])

  return ref
}
