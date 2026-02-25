'use client';

import { useState } from 'react';
import { Download, FileJson, FileText, FileSpreadsheet } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

interface ExportPanelProps {
  data: any;
  filename?: string;
  onExport?: (format: string) => void;
}

const formats = [
  { id: 'json', name: 'JSON', icon: FileJson, description: 'Raw data format' },
  { id: 'markdown', name: 'Markdown', icon: FileText, description: 'Human readable' },
  { id: 'csv', name: 'CSV', icon: FileSpreadsheet, description: 'Spreadsheet compatible' },
];

export function ExportPanel({ data, filename = 'export', onExport }: ExportPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleExport = (format: string) => {
    let content: string;
    let mimeType: string;
    let extension: string;

    switch (format) {
      case 'json':
        content = JSON.stringify(data, null, 2);
        mimeType = 'application/json';
        extension = 'json';
        break;
      case 'markdown':
        content = dataToMarkdown(data);
        mimeType = 'text/markdown';
        extension = 'md';
        break;
      case 'csv':
        content = dataToCSV(data);
        mimeType = 'text/csv';
        extension = 'csv';
        break;
      default:
        return;
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.${extension}`;
    a.click();
    URL.revokeObjectURL(url);

    onExport?.(format);
    setIsOpen(false);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-4 py-2 rounded-lg bg-cyber-dark text-gray-400 hover:text-white hover:bg-layer1/10 transition-all"
      >
        <Download className="w-4 h-4" />
        Export
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute right-0 mt-2 w-56 bg-cyber-darker border border-layer1/30 rounded-lg shadow-xl z-50"
          >
            <div className="p-2">
              {formats.map((format) => {
                const Icon = format.icon;
                return (
                  <button
                    key={format.id}
                    onClick={() => handleExport(format.id)}
                    className="w-full flex items-center gap-3 p-3 rounded-lg hover:bg-cyber-dark transition-all text-left"
                  >
                    <Icon className="w-5 h-5 text-gray-400" />
                    <div>
                      <div className="font-medium">{format.name}</div>
                      <div className="text-xs text-gray-500">{format.description}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function dataToMarkdown(data: any): string {
  let md = '# Research Results\n\n';

  if (data.finding) {
    md += '## Summary\n\n';
    md += `- **Sources**: ${data.finding.sources?.length || 0}\n`;
    md += `- **Hypotheses**: ${data.finding.hypotheses?.length || 0}\n\n`;

    if (data.finding.sources?.length > 0) {
      md += '## Sources\n\n';
      data.finding.sources.forEach((source: any, i: number) => {
        md += `### ${i + 1}. ${source.title || source.url}\n`;
        md += `- **URL**: ${source.url}\n`;
        md += `- **ETS Score**: ${((source.ets_score || 0.5) * 100).toFixed(0)}%\n`;
        md += `- **Domain**: ${source.domain}\n\n`;
      });
    }

    if (data.finding.hypotheses?.length > 0) {
      md += '## Hypotheses\n\n';
      data.finding.hypotheses.forEach((hyp: any, i: number) => {
        md += `### Hypothesis ${i + 1}\n`;
        md += `${hyp.claim}\n\n`;
        md += `- **Confidence**: ${(hyp.confidence * 100).toFixed(0)}%\n`;
        md += `- **Status**: ${hyp.status}\n\n`;
      });
    }
  }

  if (data.metrics) {
    md += '## Metrics\n\n';
    md += `| Metric | Value |\n`;
    md += `|--------|-------|\n`;
    md += `| Total Time | ${(data.metrics.total_time_ms / 1000).toFixed(2)}s |\n`;
    md += `| Sources Found | ${data.metrics.sources_found || 0} |\n`;
    md += `| Hypotheses Built | ${data.metrics.hypotheses_built || 0} |\n`;
  }

  return md;
}

function dataToCSV(data: any): string {
  const sources = data.finding?.sources || [];
  const headers = ['Title', 'URL', 'Domain', 'ETS Score'];
  const rows = sources.map((s: any) => [
    s.title || '',
    s.url,
    s.domain,
    ((s.ets_score || 0.5) * 100).toFixed(0) + '%',
  ]);

  return [
    headers.join(','),
    ...rows.map((r: string[]) => r.map(cell => `"${cell}"`).join(',')),
  ].join('\n');
}
