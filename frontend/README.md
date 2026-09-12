# Şifacı AI Web

Premium med-tech landing page and animated RAG interface for Şifacı AI. Questions are sent through a same-origin Next.js route to the local Python API; the browser never calls Foundry Local directly.

## Stack

- Next.js App Router, React and TypeScript
- Tailwind CSS
- Motion for React
- Lucide React icons

## Run locally

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

The Python API must be running in a separate terminal from the project root:

```powershell
python api.py
```

The first Foundry Local model preparation can take longer than later requests.
`SIFACI_REQUEST_TIMEOUT_MS` defaults to five minutes and can be overridden in
`.env.local` when needed.

## Quality checks

```powershell
npm run lint
npm run typecheck
npm run build
```

## Component structure

```text
frontend/
├── app/
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
└── components/
    ├── Hero/
    │   ├── FloatingPill.tsx
    │   ├── Hero.tsx
    │   ├── MedicineBottle.tsx
    │   └── MedicineBox.tsx
    ├── Search/
    │   ├── MedicineSearch.tsx
    │   └── SuggestedQueries.tsx
    ├── Answer/
    │   └── MedicineAnswerCard.tsx
    ├── CapabilityGrid.tsx
    ├── Header.tsx
    ├── SafetyNotice.tsx
    └── SystemStatus.tsx
```

Additional integration files:

```text
app/api/answer/route.ts   Same-origin answer proxy to the Python API
app/api/health/route.ts   Live local database status proxy
lib/api.ts                Typed browser client
types/medicine.ts         API response contract
.env.example              Optional Python API URL override
```

The Şifacı AI medicine package, bottle and pills are built with React and CSS. They use no real pharmaceutical brand or stock imagery. Motion respects the operating system's reduced-motion preference.
