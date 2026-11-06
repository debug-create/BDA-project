# Statistical release theme - change report

## Before and after

| File | Before | After |
|---|---|---|
| `index.html` | Previous dashboard font setup | Google Fonts setup for Libre Franklin, Public Sans, Source Serif 4 and Red Hat Mono, with preconnect and preload hints |
| `src/styles.css` | Dark airport-board dashboard as the primary visual language | Light statistical-release portal, optional dark theme, report rules, dense tables, direct chart styling, responsive top navigation and motion states |
| `src/App.tsx` | Page content without report apparatus | Numbered pages, breadcrumb, figure/table wrappers, sources, findings, notes, ticker, date brush and independent-project footer attribution |
| `src/copy.ts` | Page and metric copy | Centralised captions, source labels, findings, technical notes, ticker text, brush labels and footer attribution |
| `src/Reveal.tsx` | Not present | Reusable reveal wrapper with optional one-time chart drawing |
| `src/useReveal.ts` | Not present | IntersectionObserver hook with below, visible and past states plus reduced-motion handling |
| `src/Report.tsx` | Not present | Reusable figure/table caption, source, key findings and technical-note components |

No result data file or numeric result value changed in this theme pass.

## Motion behavior

- Text, cards and table content use opacity and translate-only transitions.
- Entry distance is 14 px and the transition duration is 430 ms with `cubic-bezier(0.22, 1, 0.36, 1)`.
- Content above the viewport settles at 0.35 opacity and returns to full opacity on re-entry.
- Chart lines and bars draw once and remain visible.
- CSS view timelines enhance supported browsers; IntersectionObserver remains the fallback.
- Content is visible before JavaScript enhancement.
- `prefers-reduced-motion` removes movement and transition delay.
- Background motion pauses while the tab is hidden.

## Contrast checks

Contrast ratios are independent of viewport. Layout and overflow were checked at 360, 768 and 1280 px in both themes.

### Light theme

| Pair | Ratio | Result |
|---|---:|---|
| Navy text / paper background | 13.53:1 | AA pass |
| Muted text / paper background | 5.42:1 | AA pass |
| Blue links / paper background | 5.34:1 | AA pass |
| Signal-red marks / paper background | 6.03:1 | AA pass |
| Muted-green marks / paper background | 5.06:1 | AA pass |
| Holiday-gold marks / paper background | 3.72:1 | Non-text AA pass |

### Dark theme

| Pair | Ratio | Result |
|---|---:|---|
| Light text / navy background | 16.45:1 | AA pass |
| Muted text / navy background | 10.05:1 | AA pass |
| Blue links / navy background | 7.90:1 | AA pass |
| Signal-red marks / navy background | 7.07:1 | AA pass |
| Muted-green marks / navy background | 9.61:1 | AA pass |
| Holiday-gold marks / navy background | 9.79:1 | AA pass |

The gold token is used for holiday bands and chart marks, not body text.

## Data binding check

The selected and reference model values remain read from `public/data/model_runs.json`. Dataset, season, split, cluster and holiday values remain read from `public/data/site-data.json`. A source search found none of the selected result values typed into React components or copy files.
