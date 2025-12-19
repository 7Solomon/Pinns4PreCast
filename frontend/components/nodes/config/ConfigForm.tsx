import React, { useState } from 'react';
import { Save, Plus, Trash2, ChevronRight, ChevronDown } from 'lucide-react';

// Helper to determine input type
const getFieldType = (prop: any) => {
    if (prop.$ref || (prop.allOf && prop.allOf[0].$ref)) return 'object'; // Nested Pydantic Model
    if (prop.type === 'integer' || prop.type === 'number') return 'number';
    if (prop.type === 'boolean') return 'boolean';
    if (prop.type === 'array') return 'array';
    if (prop.type === 'object') return 'dict'; // Generic Dictionary
    return 'text';
};

// Simple Check Icon Component
const CheckIcon = () => (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M10 3L4.5 8.5L2 6" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
);

const FormField = ({ name, schema, value, onChange, definitions }: any) => {
    const title = schema.title || name;
    const type = getFieldType(schema);
    const [isExpanded, setIsExpanded] = useState(true);

    const description = schema.description;

    // --- NESTED OBJECTS (Sub-Models) ---
    if (type === 'object') {
        const refName = (schema.$ref || schema.allOf[0].$ref).split('/').pop();
        const refSchema = definitions[refName];

        return (
            <div className="mb-6 bg-slate-950/50 border border-slate-800 rounded-lg overflow-hidden">
                <div
                    className="bg-slate-900 px-4 py-3 flex items-center justify-between cursor-pointer hover:bg-slate-800/80 transition-colors"
                    onClick={() => setIsExpanded(!isExpanded)}
                >
                    <div className="flex items-center gap-2">
                        {isExpanded ? <ChevronDown size={16} className="text-blue-400" /> : <ChevronRight size={16} className="text-slate-500" />}
                        <span className="text-sm font-semibold text-slate-200">{title}</span>
                    </div>
                    <span className="text-[10px] text-slate-500 font-mono uppercase">Section</span>
                </div>

                {isExpanded && (
                    <div className="p-4 border-t border-slate-800 space-y-4">
                        {Object.keys(refSchema.properties).map((propKey) => (
                            <FormField
                                key={propKey}
                                name={propKey}
                                schema={refSchema.properties[propKey]}
                                definitions={definitions}
                                value={value?.[propKey] ?? refSchema.properties[propKey].default}
                                onChange={(val: any) => onChange({ ...value, [propKey]: val })}
                            />
                        ))}
                    </div>
                )}
            </div>
        );
    }

    // --- ARRAYS ---
    if (type === 'array') {
        const items = value || [];
        return (
            <div className="mb-5 p-4 bg-slate-950 rounded-lg border border-slate-800">
                <div className="flex justify-between items-center mb-3">
                    <label className="text-sm font-medium text-slate-300">{title}</label>
                    <button
                        onClick={() => onChange([...items, 0])}
                        className="text-xs flex items-center gap-1 bg-blue-500/10 text-blue-400 px-2 py-1 rounded hover:bg-blue-500/20 transition-colors"
                    >
                        <Plus size={12} /> Add
                    </button>
                </div>

                <div className="space-y-2">
                    {items.length === 0 && <div className="text-xs text-slate-600 italic">No items defined</div>}
                    {items.map((item: any, idx: number) => (
                        <div key={idx} className="flex gap-2 group">
                            <input
                                type="number"
                                className="flex-1 bg-slate-900 border border-slate-700 rounded-md px-3 py-2 text-sm text-slate-200 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all"
                                value={item}
                                onChange={(e) => {
                                    const newItems = [...items];
                                    newItems[idx] = Number(e.target.value);
                                    onChange(newItems);
                                }}
                            />
                            <button
                                onClick={() => onChange(items.filter((_: any, i: number) => i !== idx))}
                                className="p-2 text-slate-500 hover:text-red-400 hover:bg-red-400/10 rounded transition-colors"
                            >
                                <Trash2 size={16} />
                            </button>
                        </div>
                    ))}
                </div>
            </div>
        );
    }

    // --- STANDARD INPUTS ---
    return (
        <div className="mb-4">
            <div className="flex justify-between items-baseline mb-1.5">
                <label className="block text-sm font-medium text-slate-300">{title}</label>
                {type !== 'boolean' && description && (
                    <span className="text-[10px] text-slate-500 truncate max-w-[200px]" title={description}>{description}</span>
                )}
            </div>

            {type === 'boolean' ? (
                <label className="flex items-center gap-3 p-3 bg-slate-950 border border-slate-800 rounded-lg cursor-pointer hover:border-slate-700 transition-colors">
                    <div className={`w-5 h-5 rounded flex items-center justify-center border transition-all ${value ? 'bg-blue-500 border-blue-500' : 'bg-slate-900 border-slate-600'}`}>
                        {value && <CheckIcon />}
                    </div>
                    <input
                        type="checkbox"
                        checked={value}
                        onChange={(e) => onChange(e.target.checked)}
                        className="hidden"
                    />
                    <span className="text-sm text-slate-400 select-none">{value ? 'Enabled' : 'Disabled'}</span>
                </label>
            ) : (
                <input
                    type={type === 'number' ? 'number' : 'text'}
                    className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2.5 text-sm text-slate-200 placeholder-slate-600 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none transition-all"
                    value={value ?? ''}
                    placeholder={description}
                    onChange={(e) => {
                        const val = type === 'number' ? Number(e.target.value) : e.target.value;
                        onChange(val);
                    }}
                />
            )}
        </div>
    );
};

export const ConfigForm = ({ schema, initialData, onSave }: any) => {
    const [formData, setFormData] = useState(initialData || {});

    // Pydantic usually puts definitions in $defs or definitions
    const definitions = schema.$defs || schema.definitions || {};

    return (
        <div className="flex flex-col h-full">
            <div className="flex-1 overflow-y-auto space-y-1 pr-2">
                {Object.keys(schema.properties || {}).map((key) => (
                    <FormField
                        key={key}
                        name={key}
                        schema={schema.properties[key]}
                        definitions={definitions}
                        value={formData[key] ?? schema.properties[key].default}
                        onChange={(val: any) => setFormData({ ...formData, [key]: val })}
                    />
                ))}
            </div>

            <div className="mt-8 pt-6 border-t border-slate-800 flex justify-end sticky bottom-0 bg-slate-900 pb-2">
                <button
                    onClick={() => onSave(formData)}
                    className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2.5 rounded-lg text-sm font-semibold shadow-lg shadow-blue-900/20 flex items-center gap-2 transition-all hover:scale-105 active:scale-95"
                >
                    <Save size={18} /> Save Changes
                </button>
            </div>
        </div>
    );
};
