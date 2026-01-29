"use client";

import { motion } from "framer-motion";
import NeonCard from "@/components/NeonCard";

export default function Slide08ClassificationResults() {
  const results = [
    { model: "Random Forest", f1: 0.927, mcc: 0.890, auc: 0.961, winner: true },
    { model: "Gradient Boosting", f1: 0.925, mcc: 0.886, auc: 0.961, winner: false },
    { model: "SVM (RBF)", f1: 0.874, mcc: 0.811, auc: 0.935, winner: false },
  ];

  return (
    <div className="relative flex-1 flex flex-col p-8 overflow-hidden">

      <motion.h2
        initial={{ opacity: 0, x: -50 }}
        animate={{ opacity: 1, x: 0 }}
        className="relative z-10 text-5xl font-display font-bold text-neon-primary mb-2 leading-tight"
      >
        Classification Results
      </motion.h2>
      
      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="relative z-10 text-lg text-neon-muted mb-6"
      >
        Comparaison des performances
      </motion.p>

      <div className="relative z-10 flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 min-h-0">
        <div className="flex flex-col gap-4 min-h-0">
          <NeonCard delay={0.4} glow="cyan" title="Results" className="shrink-0">
            <div className="mt-4 overflow-x-auto custom-scrollbar">
              <table className="w-full min-w-[400px]">
                <thead>
                  <tr className="text-neon-muted text-sm border-b border-neon-muted/20">
                    <th className="text-left py-1.5 whitespace-nowrap">Model</th>
                    <th className="text-right py-1.5 whitespace-nowrap">F1</th>
                    <th className="text-right py-1.5 whitespace-nowrap">MCC</th>
                    <th className="text-right py-1.5 whitespace-nowrap">AUC</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <motion.tr
                      key={r.model}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.6 + i * 0.15 }}
                      className={`border-b border-neon-muted/10 ${r.winner ? "bg-neon-primary/10" : ""}`}
                    >
                      <td className={`py-2 text-sm ${r.winner ? "text-neon-primary font-bold" : "text-neon-text"}`}>
                        {r.winner && "★ "}{r.model}
                      </td>
                      <td className="py-2 text-right font-mono text-neon-primary text-sm">{r.f1.toFixed(3)}</td>
                      <td className="py-2 text-right font-mono text-neon-primary text-sm">{r.mcc.toFixed(3)}</td>
                      <td className="py-2 text-right font-mono text-neon-primary text-sm">{r.auc.toFixed(3)}</td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </NeonCard>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.2 }}
            className="p-4 bg-neon-primary/10 border border-neon-primary/30 rounded-xl shrink-0"
          >
            <h4 className="text-neon-primary font-bold text-lg mb-1">Winner: Random Forest</h4>
            <p className="text-neon-muted text-sm">All classifiers achieve &gt;87% F1-score</p>
          </motion.div>
        </div>

        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8 }}
          className="rounded-xl overflow-hidden border-2 border-neon-primary/30 bg-neon-bg/40 flex items-center justify-center p-4 min-h-0"
        >
          <img
            src="/figures/classification_comparison.png"
            alt="Classification Comparison"
            className="max-w-full max-h-full object-contain"
          />
        </motion.div>
      </div>
    </div>
  );
}
