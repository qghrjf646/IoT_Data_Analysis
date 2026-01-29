"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";
import { TreeDeciduous, Circle, Layers } from "lucide-react";

export default function Slide05AnomalyMethods() {
  const methods = [
    {
      name: "Isolation Forest",
      icon: TreeDeciduous,
      desc: "Tree-based isolation via random partitioning",
      color: "cyan" as const,
    },
    {
      name: "One-Class SVM",
      icon: Circle,
      desc: "Kernel-based boundary in feature space",
      color: "violet" as const,
    },
    {
      name: "Local Outlier Factor",
      icon: Layers,
      desc: "Local density deviation detection",
      color: "magenta" as const,
    },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-7xl font-display font-bold text-neon-primary mb-2"
      >
        Anomaly Detection
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-xl text-neon-muted mb-4"
      >
        Méthodes non supervisées
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="relative z-10 mb-8 p-4 bg-neon-accent/10 border border-neon-accent/30 rounded-xl inline-block"
      >
        <span className="text-neon-accent font-bold">Trained on benign traffic only</span>
        <span className="text-neon-muted"> — Détecte les déviations du comportement normal</span>
      </motion.div>

      <div className="relative z-10 flex-1 grid grid-cols-3 gap-8">
        {methods.map((method, i) => (
          <NeonCard 
            key={method.name} 
            delay={0.5 + i * 0.2} 
            glow={method.color}
            className="flex flex-col items-center text-center h-full"
          >
            <motion.div
              initial={{ scale: 0, rotate: -20 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ delay: 0.8 + i * 0.2, type: "spring", stiffness: 200 }}
              className="mb-6"
            >
              <method.icon 
                size={64} 
                strokeWidth={1.5}
                className={`
                  ${method.color === "cyan" ? "text-neon-primary" : ""}
                  ${method.color === "magenta" ? "text-neon-secondary" : ""}
                  ${method.color === "violet" ? "text-neon-accent" : ""}
                `}
              />
            </motion.div>
            <h3 className="text-2xl font-display font-bold text-neon-text mb-4">
              {method.name}
            </h3>
            <p className="text-neon-muted text-lg">{method.desc}</p>
          </NeonCard>
        ))}
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.3 }}
        className="relative z-10 mt-8 p-4 bg-neon-bg/60 border border-neon-primary/30 rounded-xl"
      >
        <span className="text-neon-primary font-bold">Evaluation Metric: </span>
        <span className="text-neon-text">AUPRC (Area Under Precision-Recall Curve) — robuste pour la détection imbalanced</span>
      </motion.div>
    </div>
  );
}
