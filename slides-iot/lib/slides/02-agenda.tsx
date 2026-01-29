"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import { Database, Search, Shield, Swords, Lightbulb } from "lucide-react";

export default function Slide02Agenda() {
  const items = [
    { icon: Database, text: "Dataset Overview and Exploration", color: "cyan" as const },
    { icon: Search, text: "Anomaly Detection (Unsupervised)", color: "violet" as const },
    { icon: Shield, text: "Classification (Supervised)", color: "cyan" as const },
    { icon: Swords, text: "Adversarial Machine Learning", color: "magenta" as const },
    { icon: Lightbulb, text: "Recommendations", color: "violet" as const },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-primary mb-4"
      >
        Agenda
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-2xl text-neon-muted mb-12"
      >
        Plan de présentation
      </motion.p>

      <div className="relative z-10 flex-1 flex flex-col justify-center gap-6 max-w-4xl">
        {items.map((item, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 + i * 0.15 }}
            className="flex items-center gap-6 group"
          >
            <div className={`
              w-16 h-16 rounded-xl flex items-center justify-center
              bg-neon-bg/60 border-2 transition-all duration-300
              ${item.color === "cyan" ? "border-neon-primary/50 group-hover:shadow-neon-cyan" : ""}
              ${item.color === "magenta" ? "border-neon-secondary/50 group-hover:shadow-neon-magenta" : ""}
              ${item.color === "violet" ? "border-neon-accent/50 group-hover:shadow-neon-violet" : ""}
            `}>
              <item.icon 
                size={32} 
                className={`
                  ${item.color === "cyan" ? "text-neon-primary" : ""}
                  ${item.color === "magenta" ? "text-neon-secondary" : ""}
                  ${item.color === "violet" ? "text-neon-accent" : ""}
                `}
              />
            </div>
            <div className="flex items-center gap-4">
              <span className="text-3xl font-display font-bold text-neon-muted">{i + 1}.</span>
              <span className="text-2xl text-neon-text font-mono">{item.text}</span>
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.5 }}
        className="relative z-10 mt-8"
      >
        <NeonCard glow="magenta" className="inline-block">
          <div className="flex items-center gap-4">
            <span className="text-neon-secondary font-bold">Objectif:</span>
            <span className="text-neon-text">Évaluer les méthodes ML pour IIoT intrusion detection et robustesse adversariale</span>
          </div>
        </NeonCard>
      </motion.div>
    </div>
  );
}
