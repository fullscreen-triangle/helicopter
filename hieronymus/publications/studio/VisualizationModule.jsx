import React, { useEffect } from 'react';
import * as d3 from 'd3';
import { Canvas } from 'react-three-fiber'; // For 3D charts

const VisualizationModule = ({ data, imageData }) => {
    const { distances, organelleSizes, cellDensity, intensityOverTime, organellePositions } = data;

    useEffect(() => {
        // Clear previous charts
        d3.selectAll('.chart').selectAll('*').remove();

        // 1. Histogram of Organelle Distances
        const histogram = d3.select('#histogram');
        const histogramMargin = { top: 20, right: 30, bottom: 30, left: 40 };
        const histogramWidth = 400 - histogramMargin.left - histogramMargin.right;
        const histogramHeight = 300 - histogramMargin.top - histogramMargin.bottom;

        const histogramSvg = histogram.append('svg')
            .attr('width', histogramWidth + histogramMargin.left + histogramMargin.right)
            .attr('height', histogramHeight + histogramMargin.top + histogramMargin.bottom)
            .append('g')
            .attr('transform', `translate(${histogramMargin.left}, ${histogramMargin.top})`);

        const x = d3.scaleLinear()
            .domain([0, d3.max(distances)])
            .range([0, histogramWidth]);

        const bins = d3.histogram()
            .value(d => d)
            .domain(x.domain())
            .thresholds(x.ticks(10))(distances);

        const y = d3.scaleLinear()
            .domain([0, d3.max(bins, d => d.length)])
            .range([histogramHeight, 0]);

        histogramSvg.selectAll('rect')
            .data(bins)
            .enter().append('rect')
            .attr('x', 1)
            .attr('transform', d => `translate(${x(d.x0)}, ${y(d.length)})`)
            .attr('width', d => x(d.x1) - x(d.x0) - 1)
            .attr('height', d => histogramHeight - y(d.length))
            .style('fill', 'steelblue');

        histogramSvg.append('g')
            .attr('transform', `translate(0, ${histogramHeight})`)
            .call(d3.axisBottom(x));

        histogramSvg.append('g')
            .call(d3.axisLeft(y));

        // 2. Box Plot of Organelle Sizes
        const boxPlot = d3.select('#boxplot');
        const boxPlotMargin = { top: 20, right: 30, bottom: 30, left: 40 };
        const boxPlotWidth = 400 - boxPlotMargin.left - boxPlotMargin.right;
        const boxPlotHeight = 300 - boxPlotMargin.top - boxPlotMargin.bottom;

        const boxPlotSvg = boxPlot.append('svg')
            .attr('width', boxPlotWidth + boxPlotMargin.left + boxPlotMargin.right)
            .attr('height', boxPlotHeight + boxPlotMargin.top + boxPlotMargin.bottom)
            .append('g')
            .attr('transform', `translate(${boxPlotMargin.left}, ${boxPlotMargin.top})`);

        const boxPlotData = [organelleSizes]; // Assuming organelleSizes is an array of sizes

        const box = d3.box()
            .whiskers(iqr(1.5))
            .height(boxPlotHeight)
            .width(boxPlotWidth)
            .domain([0, d3.max(organelleSizes)]);

        boxPlotSvg.datum(boxPlotData).call(box);

        // 3. Heatmap of Cell Density
        const heatmap = d3.select('#heatmap');
        const heatmapMargin = { top: 20, right: 30, bottom: 30, left: 40 };
        const heatmapWidth = 400 - heatmapMargin.left - heatmapMargin.right;
        const heatmapHeight = 300 - heatmapMargin.top - heatmapMargin.bottom;

        const heatmapSvg = heatmap.append('svg')
            .attr('width', heatmapWidth + heatmapMargin.left + heatmapMargin.right)
            .attr('height', heatmapHeight + heatmapMargin.top + heatmapMargin.bottom)
            .append('g')
            .attr('transform', `translate(${heatmapMargin.left}, ${heatmapMargin.top})`);

        const colorScale = d3.scaleSequential(d3.interpolateBlues)
            .domain([0, d3.max(cellDensity)]);

        // Assuming cellDensity is a 2D array
        heatmapSvg.selectAll('rect')
            .data(cellDensity)
            .enter().append('rect')
            .attr('x', (d, i) => i % Math.sqrt(cellDensity.length) * (heatmapWidth / Math.sqrt(cellDensity.length)))
            .attr('y', (d, i) => Math.floor(i / Math.sqrt(cellDensity.length)) * (heatmapHeight / Math.sqrt(cellDensity.length)))
            .attr('width', heatmapWidth / Math.sqrt(cellDensity.length))
            .attr('height', heatmapHeight / Math.sqrt(cellDensity.length))
            .style('fill', d => colorScale(d));

        // 4. Line Chart of Intensity Over Time
        const lineChart = d3.select('#linechart');
        const lineChartMargin = { top: 20, right: 30, bottom: 30, left: 40 };
        const lineChartWidth = 400 - lineChartMargin.left - lineChartMargin.right;
        const lineChartHeight = 300 - lineChartMargin.top - lineChartMargin.bottom;

        const lineChartSvg = lineChart.append('svg')
            .attr('width', lineChartWidth + lineChartMargin.left + lineChartMargin.right)
            .attr('height', lineChartHeight + lineChartMargin.top + lineChartMargin.bottom)
            .append('g')
            .attr('transform', `translate(${lineChartMargin.left}, ${lineChartMargin.top})`);

        const line = d3.line()
            .x((d, i) => xScale(i))
            .y(d => yScale(d.intensity));

        lineChartSvg.append('path')
            .datum(intensityOverTime)
            .attr('fill', 'none')
            .attr('stroke', 'steelblue')
            .attr('stroke-width', 1.5)
            .attr('d', line);

        // 5. Scatter Plot of Organelle Positions
        const scatterPlot = d3.select('#scatterplot');
        const scatterPlotMargin = { top: 20, right: 30, bottom: 30, left: 40 };
        const scatterPlotWidth = 400 - scatterPlotMargin.left - scatterPlotMargin.right;
        const scatterPlotHeight = 300 - scatterPlotMargin.top - scatterPlotMargin.bottom;

        const scatterPlotSvg = scatterPlot.append('svg')
            .attr('width', scatterPlotWidth + scatterPlotMargin.left + scatterPlotMargin.right)
            .attr('height', scatterPlotHeight + scatterPlotMargin.top + scatterPlotMargin.bottom)
            .append('g')
            .attr('transform', `translate(${scatterPlotMargin.left}, ${scatterPlotMargin.top})`);

        scatterPlotSvg.selectAll('circle')
            .data(organellePositions)
            .enter().append('circle')
            .attr('cx', d => xScale(d.x)) // Assuming organellePositions has x and y properties
            .attr('cy', d => yScale(d.y))
            .attr('r', 5)
            .style('fill', 'orange');

        // 3D Visualizations
        // Implement 3D visualizations using react-three-fiber here

    }, [data]);

    return (
        <div className="visualization-module">
            <h2>Microscopy Data Visualizations</h2>
            <div className="charts">
                <div id="histogram" className="chart"></div>
                <div id="boxplot" className="chart"></div>
                <div id="heatmap" className="chart"></div>
                <div id="linechart" className="chart"></div>
                <div id="scatterplot" className="chart"></div>
            </div>
            <Canvas>
                {/* 3D visualizations go here */}
            </Canvas>
            <div className="image-annotations">
                <img src={imageData} alt="Microscopy Cell" />
                {/* Add annotations overlay logic */}
            </div>
        </div>
    );
};

export default VisualizationModule;
