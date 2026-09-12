import { CapabilityGrid } from "@/components/CapabilityGrid";
import { Hero } from "@/components/Hero/Hero";
import { SafetyNotice } from "@/components/SafetyNotice";

export default function Home() {
  return (
    <main className="min-h-screen overflow-hidden">
      <Hero />
      <CapabilityGrid />
      <SafetyNotice />
    </main>
  );
}
