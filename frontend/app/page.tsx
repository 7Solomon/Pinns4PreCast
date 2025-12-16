"use client";

import dynamic from 'next/dynamic';

const Editor = dynamic(() => import('@/components/FlowEditor'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-screen flex items-center justify-center bg-slate-950 text-white">
      Loading DeepONet Studio...
    </div>
  )
});

export default function Home() {
  return (
    <main className="w-screen h-screen overflow-hidden bg-slate-950">
      <Editor />
    </main>
  );
}
