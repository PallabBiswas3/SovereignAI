import React, { useEffect, useRef } from "react";
import * as d3 from "d3";
import {
  Simulation,
  SimulationLinkDatum,
  SimulationNodeDatum
} from "d3-force";
import { GraphData, Node, Link } from "../types";

/* ---------------------------------- */
/* Extend D3 Types                    */
/* ---------------------------------- */

interface GraphNode extends SimulationNodeDatum, Node { }

interface GraphLink extends SimulationLinkDatum<GraphNode> {
  relationship: string;
}

/* ---------------------------------- */
/* Component                          */
/* ---------------------------------- */

interface GraphCanvasProps {
  data: GraphData;
  onNodeClick: (node: Node) => void;
}

const GraphCanvas: React.FC<GraphCanvasProps> = ({
  data,
  onNodeClick
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);

  useEffect(() => {
    if (!svgRef.current || data.nodes.length === 0) return;

    const container = svgRef.current.parentElement;
    let width = container?.clientWidth || 800;
    let height = container?.clientHeight || 600;

    const updateDimensions = () => {
      width = container?.clientWidth || 800;
      height = container?.clientHeight || 600;

      const svg = d3.select(svgRef.current);
      svg
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", `0 0 ${width} ${height}`);
    };

    // Initial dimensions
    updateDimensions();

    // Add resize observer
    const resizeObserver = new ResizeObserver(updateDimensions);
    if (container) {
      resizeObserver.observe(container);
    }

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const g = svg.append("g");

    /* ---------------------------------- */
    /* Zoom                               */
    /* ---------------------------------- */

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => {
        g.attr("transform", event.transform.toString());
      });

    svg.call(zoom);

    /* ---------------------------------- */
    /* Simulation                         */
    /* ---------------------------------- */

    const simulation: Simulation<GraphNode, GraphLink> =
      d3.forceSimulation<GraphNode>(data.nodes as GraphNode[])
        .force(
          "link",
          d3
            .forceLink<GraphNode, GraphLink>(
              data.links as GraphLink[]
            )
            .id((d) => d.id)
            .distance(150)
        )
        .force("charge", d3.forceManyBody().strength(-400))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide<GraphNode>().radius(60));


    /* ---------------------------------- */
    /* Links                              */
    /* ---------------------------------- */

    const link = g
      .append("g")
      .attr("stroke", "#444")
      .attr("stroke-opacity", 0.6)
      .selectAll("line")
      .data(data.links as GraphLink[])
      .join("line")
      .attr("stroke-width", 1.5);

    const linkText = g
      .append("g")
      .selectAll("text")
      .data(data.links as GraphLink[])
      .join("text")
      .attr("font-size", "8px")
      .attr("fill", "#888")
      .attr("text-anchor", "middle")
      .text((d) => d.relationship);

    /* ---------------------------------- */
    /* Nodes                              */
    /* ---------------------------------- */

    const node = g
      .append("g")
      .selectAll<SVGGElement, GraphNode>("g")
      .data(data.nodes as GraphNode[])
      .join("g")
      .call(drag(simulation))
      .on("click", (_, d) => onNodeClick(d));

    const nodeColors: Record<string, string> = {
      Paper: "#3b82f6",
      Author: "#ef4444",
      Concept: "#10b981",
      Method: "#f59e0b",
      Result: "#8b5cf6"
    };

    node
      .append("circle")
      .attr("r", (d) => (d.type === "Paper" ? 12 : 8))
      .attr("fill", (d) => nodeColors[d.type] || "#ccc")
      .attr("stroke", "#000")
      .attr("stroke-width", 1.5);

    node
      .append("text")
      .attr("dy", 20)
      .attr("text-anchor", "middle")
      .attr("font-size", "10px")
      .attr("fill", "#fff")
      .attr("font-weight", "500")
      .text((d) => d.label);

    /* ---------------------------------- */
    /* Tick Updates                       */
    /* ---------------------------------- */

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => (d.source as GraphNode).x!)
        .attr("y1", (d) => (d.source as GraphNode).y!)
        .attr("x2", (d) => (d.target as GraphNode).x!)
        .attr("y2", (d) => (d.target as GraphNode).y!);

      linkText
        .attr(
          "x",
          (d) =>
            (((d.source as GraphNode).x! + (d.target as GraphNode).x!) / 2)
        )
        .attr(
          "y",
          (d) =>
            (((d.source as GraphNode).y! + (d.target as GraphNode).y!) / 2)
        );

      node.attr(
        "transform",
        (d) => `translate(${d.x},${d.y})`
      );
    });

    /* ---------------------------------- */
    /* Drag                               */
    /* ---------------------------------- */

    function drag(
      simulation: Simulation<GraphNode, GraphLink>
    ) {
      function dragstarted(
        event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>
      ) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
      }

      function dragged(
        event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>
      ) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
      }

      function dragended(
        event: d3.D3DragEvent<SVGGElement, GraphNode, GraphNode>
      ) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
      }

      return d3
        .drag<SVGGElement, GraphNode>()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended);
    }

    return () => {
      simulation.stop();
    };
  }, [data, onNodeClick]);

  return (
    <svg
      ref={svgRef}
      className="w-full h-full cursor-grab active:cursor-grabbing bg-black/40 rounded-xl"
    />
  );
};

export default GraphCanvas;