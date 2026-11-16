"""Export full-population log1p price/mileage charts for the final presentation."""
import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl

ROOT = Path(__file__).resolve().parents[1]


def export_charts(dataset: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    frame = pl.scan_parquet(dataset / 'used_cars.parquet').select('price', 'mileage').collect()
    receipt = {'rows': frame.height, 'transformation': 'ln(1 + x)', 'fields': {}}
    for field, color in [('price', '#175CD4'), ('mileage', '#407879')]:
        raw = frame[field].drop_nulls().to_numpy()
        raw = raw[np.isfinite(raw) & (raw >= 0)]
        values = np.log1p(raw)
        quantiles = np.quantile(values, [0, .25, .5, .75, 1])
        q1, median, q3 = quantiles[1:4]
        iqr = q3 - q1
        low = float(values[values >= q1 - 1.5 * iqr].min())
        high = float(values[values <= q3 + 1.5 * iqr].max())
        counts, edges = np.histogram(values, bins=60)
        fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts,
                              width=np.diff(edges), marker_color=color))
        fig.update_layout(template='plotly_white', width=750, height=458,
                          font=dict(family='Arial', size=23, color='#08284F'),
                          margin=dict(l=100, r=20, t=20, b=110), bargap=.03,
                          xaxis_title=f'ln(1 + {field})', yaxis_title='Listings',
                          showlegend=False, paper_bgcolor='white', plot_bgcolor='white')
        fig.update_yaxes(tickformat='~s')
        fig.write_image(output / f'{field}_hist.png', scale=2)
        fig.write_html(output / f'{field}_hist.html', include_plotlyjs=True)
        box = go.Figure(go.Box(q1=[q1], median=[median], q3=[q3],
                              lowerfence=[low], upperfence=[high], orientation='h',
                              boxpoints=False, line_color=color,
                              fillcolor='rgba(23,92,212,0.20)' if field == 'price'
                              else 'rgba(64,120,121,0.20)'))
        box.update_layout(template='plotly_white', width=754, height=147,
                          font=dict(family='Arial', size=22, color='#08284F'),
                          margin=dict(l=40, r=20, t=5, b=52), showlegend=False,
                          xaxis=dict(range=[float(quantiles[0]), float(quantiles[-1])]),
                          yaxis=dict(showticklabels=False), paper_bgcolor='white')
        box.write_image(output / f'{field}_box.png', scale=2)
        box.write_html(output / f'{field}_box.html', include_plotlyjs=True)
        receipt['fields'][field] = {'n': len(raw), 'raw_summary': np.quantile(raw, [0, .25, .5, .75, 1]).tolist(),
                                    'log1p_summary': quantiles.tolist(), 'whiskers': [low, high],
                                    'raw_skewness': float(pl.Series(raw).skew()),
                                    'log1p_skewness': float(pl.Series(values).skew())}
    (output / 'chart_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'DATA/CAR_DATA_FINAL')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/slide13_update/charts')
    args = parser.parse_args()
    print(json.dumps(export_charts(args.dataset, args.output)), flush=True)
