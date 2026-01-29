"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/slides/1");
  }, [router]);

  return (
    <div className="w-screen h-screen bg-neon-bg flex items-center justify-center">
      <div className="text-neon-primary animate-pulse">Chargement...</div>
    </div>
  );
}
