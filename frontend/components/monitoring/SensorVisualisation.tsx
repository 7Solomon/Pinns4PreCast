import React, { useState, useMemo, useEffect } from 'react';
import { Eye, Thermometer, Activity, Table } from 'lucide-react';

export interface SensorData {
    epoch: number;
    temp_csv: string;
    alpha_csv: string;
    message: string;
}

interface SensorPoint {
    time_s: number;
    time_h: number;
    temp: number;
    alpha: number;
    sensor_id: number;
}

export default function SensorVisualization({
    sensorData,
    epoch
}: {
    sensorData: SensorData | null;
    epoch: number | null;
}) {
    const [activeTab, setActiveTab] = useState<'temp' | 'alpha'>('temp');
    const [parsedData, setParsedData] = useState<{ temp: SensorPoint[][], alpha: SensorPoint[][] }>({
        temp: [],
        alpha: []
    });

    // Parse CSV data
    useEffect(() => {
        if (!sensorData) return;

        const parseCSV = (csv: string): SensorPoint[][] => {
            const lines = csv.trim().split('\n').slice(1); // Skip header
            const points: SensorPoint[][] = [];

            lines.forEach((line, timeIdx) => {
                if (!line.trim()) return;
                const values = line.split(',').map(v => parseFloat(v));
                if (values.length < 12) return;

                const time_s = values[0];
                const time_h = values[1];
                const sensors: SensorPoint[] = [];

                for (let i = 1; i <= 10; i++) {
                    sensors.push({
                        time_s,
                        time_h,
                        temp: values[2 + i - 1],  // T1_Temp -> T10_Temp
                        alpha: 0, // Will be filled for alpha tab
                        sensor_id: i
                    });
                }
                points.push(sensors);
            });

            return points;
        };

        const tempPoints = parseCSV(sensorData.temp_csv);
        const alphaPoints = parseCSV(sensorData.alpha_csv);

        // Fill alpha values into temp structure
        const combinedTemp: SensorPoint[][] = tempPoints.map((sensors, timeIdx) => {
            return sensors.map((sensor, sensorIdx) => ({
                ...sensor,
                alpha: alphaPoints[timeIdx]?.[sensorIdx]?.temp || 0 // Reuse temp parsing logic for alpha
            }));
        });

        setParsedData({ temp: tempPoints, alpha: combinedTemp });
    }, [sensorData]);

    const latestData = useMemo(() => {
        if (!parsedData[activeTab].length) return null;
        return parsedData[activeTab][parsedData[activeTab].length - 1];
    }, [parsedData, activeTab]);

    const timeSeries = useMemo(() => {
        return parsedData[activeTab].map((sensors, idx) => ({
            time: sensors[0].time_h,
            sensors: sensors.map(s => activeTab === 'temp' ? s.temp : s.alpha)
        }));
    }, [parsedData, activeTab]);

    if (!sensorData || !epoch) {
        return (
            <div className="h-64 flex items-center justify-center bg-gradient-to-br from-slate-900/50 to-slate-800/50 border-2 border-dashed border-slate-700 rounded-xl">
                <div className="text-center text-slate-500">
                    <Eye className="w-12 h-12 mx-auto mb-3 opacity-30" />
                    <p className="text-sm">Waiting for sensor data...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 h-80 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <Eye size={16} className="text-cyan-400" />
                    Sensors Epoch {epoch}
                </h4>
                <div className="flex gap-1">
                    <button
                        onClick={() => setActiveTab('temp')}
                        className={`px-3 py-1 text-xs rounded-lg font-medium transition-all ${activeTab === 'temp'
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                            }`}
                    >
                        <Thermometer size={12} className="inline mr-1" />
                        Temperature
                    </button>
                    <button
                        onClick={() => setActiveTab('alpha')}
                        className={`px-3 py-1 text-xs rounded-lg font-medium transition-all ${activeTab === 'alpha'
                            ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                            }`}
                    >
                        <Activity size={12} className="inline mr-1" />
                        Alpha
                    </button>
                </div>
            </div>

            {/* Latest Data Table */}
            <div className="flex-1 overflow-hidden">
                {latestData ? (
                    <div className="h-full overflow-y-auto custom-scrollbar">
                        <table className="w-full text-xs">
                            <thead>
                                <tr className="bg-slate-800/50 text-slate-400 sticky top-0">
                                    <th className="p-2 text-left font-mono">Time</th>
                                    {latestData.map((sensor, idx) => (
                                        <th key={idx} className="p-2 text-center font-mono">
                                            T{idx + 1}
                                        </th>
                                    ))}
                                </tr>
                            </thead>
                            <tbody>
                                <tr className="border-t border-slate-700">
                                    <td className="p-2 font-mono text-slate-300">
                                        t={latestData[0].time_h.toFixed(1)}h
                                    </td>
                                    {latestData.map((sensor, idx) => (
                                        <td
                                            key={idx}
                                            className="p-2 text-center font-mono"
                                            style={{
                                                color: activeTab === 'temp'
                                                    ? sensor.temp > 23 ? '#22c55e' : '#f59e0b'
                                                    : sensor.alpha > 0 ? '#8b5cf6' : '#ec4899'
                                            }}
                                        >
                                            {activeTab === 'temp'
                                                ? sensor.temp.toFixed(2)
                                                : sensor.alpha.toFixed(4)
                                            }
                                        </td>
                                    ))}
                                </tr>
                            </tbody>
                        </table>

                        {/* Time series preview */}
                        {timeSeries.length > 1 && (
                            <div className="mt-3 pt-3 border-t border-slate-800">
                                <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
                                    <div className="w-3 h-3 rounded-full bg-gradient-to-r from-cyan-400 to-blue-500"></div>
                                    Sensor T1 {activeTab === 'temp' ? 'Temp' : 'Alpha'} trend
                                </div>
                                <div className="w-full h-20 bg-slate-850 rounded-lg p-2 flex items-end gap-1 overflow-hidden">
                                    {timeSeries.slice(-20).map((point, idx) => (
                                        <div
                                            key={idx}
                                            className="flex-1 bg-gradient-to-t from-slate-700 to-cyan-500 rounded transition-all"
                                            style={{
                                                height: `${Math.max(0, (point.sensors[0] / Math.max(...timeSeries.map(p => p.sensors[0]))) * 100)}%`
                                            }}
                                            title={`${point.time.toFixed(1)}h: ${point.sensors[0].toFixed(activeTab === 'temp' ? 2 : 4)}`}
                                        />
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                ) : (
                    <div className="h-full flex items-center justify-center text-slate-500">
                        Parsing sensor data...
                    </div>
                )}
            </div>
        </div>
    );
}
