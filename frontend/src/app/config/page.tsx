'use client';

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { configApi } from '@/lib/api';
import { Sidebar } from '@/components/layout/Sidebar';
import { Header } from '@/components/layout/Header';
import { Save, RotateCcw, Check } from 'lucide-react';
import { motion } from 'framer-motion';

export default function ConfigPage() {
  const queryClient = useQueryClient();
  const [selectedModule, setSelectedModule] = useState<string | null>(null);
  const [localConfig, setLocalConfig] = useState<Record<string, any>>({});
  const [hasChanges, setHasChanges] = useState(false);

  const { data: config, isLoading } = useQuery({
    queryKey: ['config'],
    queryFn: () => configApi.get(),
  });

  const { data: schema } = useQuery({
    queryKey: ['config-schema'],
    queryFn: () => configApi.getSchema(),
  });

  const { data: presets } = useQuery({
    queryKey: ['config-presets'],
    queryFn: () => configApi.getPresets(),
  });

  const updateMutation = useMutation({
    mutationFn: (config: Record<string, any>) => configApi.update(config),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['config'] });
      setHasChanges(false);
    },
  });

  const modules = config?.data ? Object.keys(config.data) : [];

  const handleConfigChange = (module: string, field: string, value: any) => {
    setLocalConfig((prev) => ({
      ...prev,
      [module]: {
        ...prev[module],
        [field]: value,
      },
    }));
    setHasChanges(true);
  };

  const handleSave = () => {
    updateMutation.mutate(localConfig);
  };

  if (isLoading) {
    return (
      <div className="flex h-screen bg-cyber-dark">
        <Sidebar />
        <div className="flex-1 flex items-center justify-center">
          <div className="animate-spin w-8 h-8 border-2 border-layer1 border-t-transparent rounded-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen bg-cyber-dark">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header title="Configuration" />
        <main className="flex-1 overflow-y-auto p-6">
          <div className="flex gap-6 h-full">
            {/* Module List */}
            <div className="w-64 bg-cyber-darker border border-layer1/30 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-4">Modules</h3>
              <div className="space-y-1">
                {modules.map((module) => (
                  <button
                    key={module}
                    onClick={() => setSelectedModule(module)}
                    className={`w-full text-left px-3 py-2 rounded-lg transition-all ${
                      selectedModule === module
                        ? 'bg-layer1/20 text-white'
                        : 'text-gray-400 hover:text-white hover:bg-layer1/10'
                    }`}
                  >
                    <span className="capitalize">{module.replace('_', ' ')}</span>
                  </button>
                ))}
              </div>
            </div>

            {/* Config Editor */}
            <div className="flex-1 bg-cyber-darker border border-layer1/30 rounded-lg p-6">
              {selectedModule ? (
                <>
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-semibold capitalize">
                      {selectedModule.replace('_', ' ')}
                    </h3>
                    <div className="flex gap-2">
                      <button
                        onClick={() => setLocalConfig({})}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-cyber-dark text-gray-400 hover:text-white transition-all"
                      >
                        <RotateCcw className="w-4 h-4" />
                        Reset
                      </button>
                      <button
                        onClick={handleSave}
                        disabled={!hasChanges || updateMutation.isPending}
                        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-layer1 text-white hover:bg-layer1/80 transition-all disabled:opacity-50"
                      >
                        {updateMutation.isPending ? (
                          <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        ) : (
                          <Save className="w-4 h-4" />
                        )}
                        Save
                      </button>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {schema?.data?.[selectedModule]?.fields?.map((field: any) => (
                      <div key={field.name} className="space-y-2">
                        <label className="text-sm text-gray-400 capitalize">
                          {field.name.replace('_', ' ')}
                        </label>
                        {field.type === 'boolean' ? (
                          <button
                            onClick={() =>
                              handleConfigChange(
                                selectedModule,
                                field.name,
                                !localConfig[selectedModule]?.[field.name] ??
                                  !config?.data?.[selectedModule]?.[field.name]
                              )
                            }
                            className={`px-4 py-2 rounded-lg font-medium transition-all ${
                              (localConfig[selectedModule]?.[field.name] ??
                                config?.data?.[selectedModule]?.[field.name])
                                ? 'bg-success/20 text-success'
                                : 'bg-cyber-dark text-gray-400'
                            }`}
                          >
                            {(localConfig[selectedModule]?.[field.name] ??
                              config?.data?.[selectedModule]?.[field.name])
                              ? 'Enabled'
                              : 'Disabled'}
                          </button>
                        ) : field.type === 'select' ? (
                          <select
                            value={
                              localConfig[selectedModule]?.[field.name] ??
                              config?.data?.[selectedModule]?.[field.name] ??
                              field.default
                            }
                            onChange={(e) =>
                              handleConfigChange(
                                selectedModule,
                                field.name,
                                e.target.value
                              )
                            }
                            className="w-full bg-cyber-dark border border-layer1/30 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-layer1"
                          >
                            {field.options?.map((opt: string) => (
                              <option key={opt} value={opt}>
                                {opt}
                              </option>
                            ))}
                          </select>
                        ) : field.type === 'number' ? (
                          <input
                            type="number"
                            value={
                              localConfig[selectedModule]?.[field.name] ??
                              config?.data?.[selectedModule]?.[field.name] ??
                              field.default
                            }
                            onChange={(e) =>
                              handleConfigChange(
                                selectedModule,
                                field.name,
                                parseFloat(e.target.value)
                              )
                            }
                            min={field.min}
                            max={field.max}
                            className="w-full bg-cyber-dark border border-layer1/30 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-layer1"
                          />
                        ) : (
                          <input
                            type="text"
                            value={
                              localConfig[selectedModule]?.[field.name] ??
                              config?.data?.[selectedModule]?.[field.name] ??
                              field.default ??
                              ''
                            }
                            onChange={(e) =>
                              handleConfigChange(
                                selectedModule,
                                field.name,
                                e.target.value
                              )
                            }
                            className="w-full bg-cyber-dark border border-layer1/30 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-layer1"
                          />
                        )}
                        {field.description && (
                          <p className="text-xs text-gray-500">{field.description}</p>
                        )}
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-500">
                  Select a module to edit its configuration
                </div>
              )}
            </div>

            {/* Presets */}
            <div className="w-64 bg-cyber-darker border border-layer1/30 rounded-lg p-4">
              <h3 className="text-sm font-medium text-gray-400 mb-4">Presets</h3>
              <div className="space-y-2">
                {presets?.data?.map((preset: any) => (
                  <button
                    key={preset.name}
                    className="w-full text-left p-3 rounded-lg bg-cyber-dark hover:bg-layer1/10 transition-all"
                  >
                    <div className="font-medium capitalize">{preset.name}</div>
                    <div className="text-xs text-gray-500 mt-1">
                      {preset.description}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
