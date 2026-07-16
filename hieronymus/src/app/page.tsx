'use client';

import dynamic from 'next/dynamic';

const LandingScene = dynamic(
  () => import('@/components/landing/LandingScene'),
  { ssr: false }
);

export default function HomePage() {
  return (
    <div className="fixed inset-0 bg-[#080c10]">
      {/* Minimal landing — only the animated Raspberry Pi camera. */}
      <LandingScene cameraOnly />
    </div>
  );
}
