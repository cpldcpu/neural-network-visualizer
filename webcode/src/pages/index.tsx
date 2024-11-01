import { useEffect, useRef, useState, Fragment } from 'react';
import { Slider } from '@/components/ui/slider';
import defaultModel from '../models/defaultModel.json';
import model1 from '../models/model1.json';
import model2 from '../models/model2.json';
import model3 from '../models/model3.json';

const ARCHITECTURE = {
  input: 64,
  hidden1: 10,
  hidden2: 10,
  output: 4 // Default value
};

// Helper to multiply matrix and vector
const matrixVectorProduct = (matrix: number[][], vector: number[]): number[] => {
  return matrix.map(row => 
    row.reduce((sum, weight, i) => sum + weight * vector[i], 0)
  );
};

// ReLU activation function
const relu = (x: number): number => Math.max(0, x);

// Layer normalization (mean=0, variance=1)
const layerNorm = (vector: number[]): number[] => {
  // Calculate mean
  const mean = vector.reduce((sum, x) => sum + x, 0) / vector.length;
  
  // Calculate variance
  const variance = vector.reduce((sum, x) => sum + (x - mean) * (x - mean), 0) / vector.length;
  
  // Add epsilon for numerical stability
  const std = Math.sqrt(variance + 1e-5);
  
  // Normalize
  return vector.map(x => (x - mean) / std);
};

// Center the input pattern
const centerInputPattern = (input: number[]): number[] => {
  const gridSize = Math.sqrt(input.length);
  const nonZeroIndices = input
    .map((value, index) => (value > 0 ? index : -1))
    .filter(index => index !== -1);

  if (nonZeroIndices.length === 0) return input;

  const xCoords = nonZeroIndices.map(index => index % gridSize);
  const yCoords = nonZeroIndices.map(index => Math.floor(index / gridSize));

  const xCenter = (Math.min(...xCoords) + Math.max(...xCoords)) / 2;
  const yCenter = (Math.min(...yCoords) + Math.max(...yCoords)) / 2;

  const xOffset = Math.floor(gridSize / 2) - Math.round(xCenter);
  const yOffset = Math.floor(gridSize / 2) - Math.round(yCenter);

  const centeredInput = new Array(input.length).fill(0);
  nonZeroIndices.forEach(index => {
    const x = index % gridSize;
    const y = Math.floor(index / gridSize);
    const newX = x + xOffset;
    const newY = y + yOffset;
    if (newX >= 0 && newX < gridSize && newY >= 0 && newY < gridSize) {
      centeredInput[newY * gridSize + newX] = input[index];
    }
  });

  return centeredInput;
};

// Modify the forwardPass function to accept centerInput as a parameter:
const forwardPass = (input: number[], weights: any, centerInput: boolean): any => {
  const processedInput = centerInput ? centerInputPattern(input) : input;
  
  // First block
  const norm1 = layerNorm(processedInput);
  // const norm1 = layerNorm(input);
  const linear1 = matrixVectorProduct(weights.hidden1, norm1);
  const act1 = linear1.map(relu);
  
  // Second block
  const norm2 = layerNorm(act1);
  const linear2 = matrixVectorProduct(weights.hidden2, norm2);
  const act2 = linear2.map(relu);
  
  // Output block
  const norm3 = layerNorm(act2);
  const output = matrixVectorProduct(weights.output, norm3);

  return {
    hidden1: act1,      // Post-ReLU
    hidden2: act2,      // Post-ReLU
    output: output      // Post-Linear (no ReLU)
  };
};

const generateRandomWeights = () => ({
  hidden1: Array.from({ length: ARCHITECTURE.hidden1 }, () => 
    Array.from({ length: ARCHITECTURE.input }, () => 
      (Math.random() * 2 - 1)
    )
  ),
  hidden2: Array.from({ length: ARCHITECTURE.hidden2 }, () => 
    Array.from({ length: ARCHITECTURE.hidden1 }, () => 
      (Math.random() * 2 - 1)
    )
  ),
  output: Array.from({ length: ARCHITECTURE.output }, () => 
    Array.from({ length: ARCHITECTURE.hidden2 }, () => 
      (Math.random() * 2 - 1)
    )
  )
});

const FuelGauge = ({ value, minValue, maxValue, isHighest = false }: { value: number, minValue: number, maxValue: number, isHighest?: boolean }) => {
  const height = 40;
  const width = 10;
  
  // Scale value between 0 and 1
  const scaledValue = maxValue === minValue ? 
    (maxValue === 0 ? 0 : 0.5) : // If all zeros show empty, otherwise show half
    (value - minValue) / (maxValue - minValue);
  
  const fillHeight = height * scaledValue;
  
  return (
    <svg width={width} height={height}>
      <rect
        x={1}
        y={0}
        width={width - 2}
        height={height}
        fill="#333"
        rx={2}
      />
      <rect
        x={2}
        y={height - fillHeight}
        width={width - 4}
        height={fillHeight}
        fill={isHighest ? '#FF009E' : '#00E5FF'}
        rx={1}
      />
    </svg>
  );
};

  // Add this at the top level of the file, after the imports
const validateWeights = (weights: any): boolean => {
  if (!weights || typeof weights !== 'object') return false;
  
  // Check structure
  const required = ['hidden1', 'hidden2', 'output'];
  if (!required.every(key => key in weights)) return false;
  
  // Check dimensions
  try {
    if (weights.hidden1.length !== ARCHITECTURE.hidden1) return false;
    if (weights.hidden1[0].length !== ARCHITECTURE.input) return false;
    
    if (weights.hidden2.length !== ARCHITECTURE.hidden2) return false;
    if (weights.hidden2[0].length !== ARCHITECTURE.hidden1) return false;
    
    if (weights.output.length > 10 || weights.output.length < 1) return false; // Ensure output neurons are between 1 and 10
    if (weights.output[0].length !== ARCHITECTURE.hidden2) return false;
    
    // Check if all values are numbers
    const allNumbers = (arr: any[]): boolean => arr.flat().every(x => typeof x === 'number' && !isNaN(x));
    if (!allNumbers(weights.hidden1) || !allNumbers(weights.hidden2) || !allNumbers(weights.output)) {
      return false;
    }

    // Classes array is optional
    if (weights.classes && (!Array.isArray(weights.classes) || weights.classes.length !== weights.output.length)) {
      return false;
    }
  } catch (e) {
    return false;
  }
  
  return true;
};

const NetworkViz = (
  { activations, config, centerInput, setCenterInput }: 
  { 
    activations: number[], 
    config: any, 
    centerInput: boolean, 
    setCenterInput: React.Dispatch<React.SetStateAction<boolean>> 
  }
) => {
  const [weightThreshold, setWeightThreshold] = useState(50);
  const [weights, setWeights] = useState(config.weights);
  const [classLabels, setClassLabels] = useState(Array(config.weights.output.length).fill("?"));
  const [layerActivations, setLayerActivations] = useState<{ hidden1: number[], hidden2: number[], output: number[] } | null>(null);

  // Update weights and class labels when config changes
  useEffect(() => {
    setWeights(config.weights);
    setClassLabels(config.classes || Array(config.weights.output.length).fill("?"));
  }, [config]);
  
  useEffect(() => {
    if (activations) {
      setLayerActivations(forwardPass(activations, weights, centerInput));
    }
  }, [activations, weights, centerInput]);

  const getLayerPositions = (layerSize: number, layerIndex: number, totalLayers: number): { x: number, y: number }[] => {
    const width = 900;  // Increased by 50%
    const height = 450; // Increased by 50%
    const padding = 40; // Padding for fuel gauges
    
    // Adjust x position to account for padding
    const layerX = padding + ((width - 2 * padding) / (totalLayers - 1)) * layerIndex;
    const positions = [];
    
    // Calculate vertical spacing based on layer size
    const verticalPadding = 30;
    const availableHeight = height - 2 * verticalPadding;
    const spacing = availableHeight / (layerSize - 1);
    
    for (let i = 0; i < layerSize; i++) {
      const y = verticalPadding + (spacing * i);
      positions.push({ x: layerX, y });
    }
    return positions;
  };

  const renderConnections = () => {
    const layers = [ARCHITECTURE.input, ARCHITECTURE.hidden1, ARCHITECTURE.hidden2, weights.output.length];
    const allPositions = layers.map((size, i) => getLayerPositions(size, i, layers.length));
    const connections: JSX.Element[] = [];
    let connectionId = 0;

    // Get source activations for each layer
    const getLayerActivations = (layerIndex: number): number[] => {
      if (!layerActivations) return new Array(layers[layerIndex]).fill(0);
      switch(layerIndex) {
        case 0: return centerInput ? centerInputPattern(activations || new Array(64).fill(0)) : (activations || new Array(64).fill(0));
        case 1: return layerActivations.hidden1;
        case 2: return layerActivations.hidden2;
        default: return [];
      }
    };

    const getVisibleProducts = (weights: number[][], sourceActs: number[]): { threshold: number, maxProduct: number } => {
      // Calculate all activation*weight products
      const products = weights.map(row => 
        row.map((w, j) => w * sourceActs[j])
      ).flat();
      
      const allProducts = products.map(Math.abs);
      const sorted = [...allProducts].sort((a, b) => a - b);
      const threshold = sorted[Math.floor((sorted.length - 1) * (weightThreshold / 100))];
      return { threshold, maxProduct: sorted[sorted.length - 1] };
    };

    const drawLayerConnections = (layer1Pos: { x: number, y: number }[], layer2Pos: { x: number, y: number }[], weights: number[][], startIndex: number) => {
      const sourceActs = getLayerActivations(startIndex);
      const { threshold, maxProduct } = getVisibleProducts(weights, sourceActs);
      
      layer1Pos.forEach((start, i) => {
        layer2Pos.forEach((end, j) => {
          const weight = weights[j][i];
          const product = weight * sourceActs[i];
          const absProduct = Math.abs(product);
          
          // Only draw connection if product is non-zero and above threshold
          if (absProduct > 0 && absProduct >= threshold) {
            // Scale thickness relative to visible products
            const scaledIntensity = (absProduct - threshold) / (maxProduct - threshold);
            const strokeWidth = 0.5 + (scaledIntensity * 2);
            const color = product > 0 ? '#00E5FF' : '#FF9E00';  // Swapped colors
            
            connections.push(
              <line
                key={`conn-${startIndex}-${connectionId++}`}
                x1={start.x}
                y1={start.y}
                x2={end.x}
                y2={end.y}
                stroke={color}
                strokeWidth={strokeWidth}
                opacity={0.5}
              />
            );
          }
        });
      });
    };

    if (weights) {
      drawLayerConnections(allPositions[0], allPositions[1], weights.hidden1, 0);
      drawLayerConnections(allPositions[1], allPositions[2], weights.hidden2, 1);
      drawLayerConnections(allPositions[2], allPositions[3], weights.output, 2);
    }

    return connections;
  };

  const renderNeurons = () => {
    const layers = [ARCHITECTURE.input, ARCHITECTURE.hidden1, ARCHITECTURE.hidden2, weights.output.length];
    const neurons: JSX.Element[] = [];
    let neuronId = 0;

    layers.forEach((layerSize, layerIndex) => {
      const positions = getLayerPositions(layerSize, layerIndex, layers.length);
      positions.forEach((pos, i) => {
        // Get activation value for this neuron
        let activation = 0;
        if (layerIndex === 0) {
          activation = centerInput ? centerInputPattern(activations || new Array(64).fill(0))[i] : (activations || new Array(64).fill(0))[i];
        } else if (layerActivations) {
          const layerName = layerIndex === 1 ? 'hidden1' : 
                           layerIndex === 2 ? 'hidden2' : 'output';
          activation = layerActivations[layerName][i] || 0;
        }

        // For output layer, determine if this is the highest activation
        const isHighestOutput = layerIndex === 3 && layerActivations?.output && 
          activation === Math.max(...layerActivations.output);

        // Draw neuron
        neurons.push(
          <g key={`neuron-${neuronId++}`}>
            {layerIndex === 0 ? (
              // Square with outline for input layer
              <rect
                x={pos.x - 4}
                y={pos.y - 4}
                width={8}
                height={8}
                fill={`rgba(255, 255, 255, ${activation})`}
                stroke="white"
                strokeWidth="1"
              />
            ) : (
              // Circle for other layers
              <circle
                cx={pos.x}
                cy={pos.y}
                r={4}
                fill="#FFFFFF"
                opacity={0.8}
              />
            )}
            {layerIndex > 0 && (
              <foreignObject
                x={pos.x + 8}
                y={pos.y - 20}
                width={layerIndex === 3 ? 50 : 12}
                height={40}
                className="flex items-center"
              >
                <div className="flex items-center gap-2">
                  <FuelGauge 
                    value={activation || 0} 
                    minValue={layerIndex === 0 ? 0 : 
                      !layerActivations ? 0 : Math.min(...(
                        layerIndex === 1 ? layerActivations.hidden1 :
                        layerIndex === 2 ? layerActivations.hidden2 :
                        layerActivations.output
                      ))}
                    maxValue={layerIndex === 0 ? 1 : 
                      !layerActivations ? 1 : Math.max(...(
                        layerIndex === 1 ? layerActivations.hidden1 :
                        layerIndex === 2 ? layerActivations.hidden2 :
                        layerActivations.output
                      ))}
                    isHighest={isHighestOutput}
                  />
                  {layerIndex === 3 && (
                    <span className="text-xs text-white ml-1 font-bold">
                      {classLabels[i]}
                    </span>
                  )}
                </div>
              </foreignObject>
            )}
          </g>
        );
      });
    });

    return neurons;
  };

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="flex items-center gap-4 text-xs text-gray-300 mb-2">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#00E5FF' }}></div>
          <span>Positive activation flow</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: '#FF9E00' }}></div>
          <span>Negative activation flow</span>
        </div>
      </div>
      <div 
        className="relative p-1 rounded-lg"
        style={{
          background: 'linear-gradient(45deg, #FF9E0011, #00E5FF11)',
          boxShadow: `
            0 0 20px 0 #FF9E0022,
            inset 0 0 20px 0 #00E5FF22
          `,
        }}
      >
        <div 
          className="absolute inset-0 rounded-lg"
          style={{
            background: 'linear-gradient(45deg, #FF9E00, #00E5FF)',
            opacity: 0.1,
            filter: 'blur(20px)',
          }}
        />
        <svg width="900" height="450" className="bg-gray-900 rounded-lg relative z-10">
          {renderConnections()}
          {renderNeurons()}
        </svg>
      </div>
      <div className="w-full mt-4 flex items-center gap-4 flex-nowrap">
        <label className="text-sm text-gray-300">Connection Threshold: {weightThreshold}%</label>
        <Slider
          value={[weightThreshold]}
          onValueChange={([value]) => setWeightThreshold(value)}
          min={0}
          max={100}
          step={1}
          className="mt-1 flex-1"
        />
        <label className="text-sm text-gray-300 flex items-center gap-2">
          <input 
            type="checkbox" 
            checked={centerInput} 
            onChange={() => setCenterInput(!centerInput)} 
          />
          Center Input
        </label>
      </div>
    </div>
  );
};

const DrawingCanvas = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [pixelData, setPixelData] = useState(new Array(64).fill(0));
  
  // Convert AVAILABLE_MODELS to a state variable
  const [availableModels, setAvailableModels] = useState([
    { name: 'Default Model', data: defaultModel },
    { name: 'Model 1', data: model1 },
    { name: 'Model 2', data: model2 },
    { name: 'Model 3', data: model3 }
  ]);

  const [networkConfig, setNetworkConfig] = useState({
    weights: defaultModel.weights,
    classes: defaultModel.classes || Array(defaultModel.weights.output.length).fill("?"),
    description: defaultModel.description || ""
  });
  
  const [errorMessage, setErrorMessage] = useState("");
  const [centerInput, setCenterInput] = useState(true);
  const [selectedModel, setSelectedModel] = useState('Default Model');
  const CANVAS_SIZE = 256;
  const GRID_SIZE = 8;
  const PIXEL_SIZE = CANVAS_SIZE / GRID_SIZE;

  // Handle weight upload and add "Custom Model" to the selection list
  const handleWeightUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!event.target.files) {
      setErrorMessage("No file selected");
      return;
    }
    const file = event.target.files[0];
    if (!file) return;
    setErrorMessage(""); // Clear any previous error

    const reader = new FileReader();
    reader.onload = (e) => {
      if (!e.target?.result) {
        setErrorMessage("Error reading file");
        return;
      }
      try {
        const json = JSON.parse(e.target.result as string);
        const { weights, classes, description } = json;
        
        if (!weights) {
          setErrorMessage("Error: No weights found in file");
          return;
        }

        if (!validateWeights(weights)) {
          setErrorMessage(
            "Invalid weight format. Expected dimensions: " +
            `Input→Hidden1: 64×10, Hidden1→Hidden2: 10×10, Hidden2→Output: 10×4`
          );
          return;
        }

        // Only use provided classes if they exist and have the correct length
        const validClasses = Array.isArray(classes) && classes.length === weights.output.length;

        setNetworkConfig({
          weights,
          classes: validClasses ? classes : Array(weights.output.length).fill("?"),
          description: description || ""
        });

        // Add "Custom Model" to availableModels if not already present
        setAvailableModels((prevModels) => {
          // Check if "Custom Model" already exists
          const customModelExists = prevModels.some(model => model.name === 'Custom Model');
          if (!customModelExists) {
            return [...prevModels, { name: 'Custom Model', data: { weights, classes, description } }];
          } else {
            // Update the existing "Custom Model" with new data
            return prevModels.map(model => 
              model.name === 'Custom Model' ? { name: 'Custom Model', data: { weights, classes, description } } : model
            );
          }
        });
      } catch (error) {
        setErrorMessage("Error parsing file. Please ensure it's valid JSON.");
      }
    };

    reader.onerror = () => {
      setErrorMessage("Error reading file");
    };

    reader.readAsText(file);
    // Reset file input
    event.target.value = '';
  };

  // Handle model selection, including "Custom Model"
  const handleModelSelect = (modelName: string) => {
    const selected = availableModels.find(model => model.name === modelName);
    if (selected) {
      setNetworkConfig({
        weights: selected.data.weights,
        classes: selected.data.classes || Array(selected.data.weights.output.length).fill("?"),
        description: selected.data.description || ""
      });
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.fillStyle = '#1A1A1A';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    
    ctx.strokeStyle = '#333333';
    for (let i = 1; i < GRID_SIZE; i++) {
      const pos = i * PIXEL_SIZE;
      ctx.beginPath();
      ctx.moveTo(pos, 0);
      ctx.lineTo(pos, CANVAS_SIZE);
      ctx.moveTo(0, pos);
      ctx.lineTo(CANVAS_SIZE, pos);
      ctx.stroke();
    }
  }, []);

  const updatePixelData = (ctx: CanvasRenderingContext2D, x: number, y: number, intensity: number) => {
    const gridX = Math.floor(x / PIXEL_SIZE);
    const gridY = Math.floor(y / PIXEL_SIZE);
    if (gridX >= 0 && gridX < GRID_SIZE && gridY >= 0 && gridY < GRID_SIZE) {
      const index = gridY * GRID_SIZE + gridX;
      const newPixelData = [...pixelData];
      newPixelData[index] = Math.min(1, newPixelData[index] + intensity);
      setPixelData(newPixelData);
      
      ctx.fillStyle = `rgba(255, 255, 255, ${newPixelData[index]})`;
      ctx.fillRect(gridX * PIXEL_SIZE, gridY * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
      
      ctx.strokeStyle = '#333333';
      ctx.strokeRect(gridX * PIXEL_SIZE, gridY * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    updatePixelData(ctx, x, y, 0.2);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#1A1A1A';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    
    ctx.strokeStyle = '#333333';
    for (let i = 0; i < GRID_SIZE; i++) {
      for (let j = 0; j < GRID_SIZE; j++) {
        ctx.strokeRect(i * PIXEL_SIZE, j * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
      }
    }
    
    setPixelData(new Array(64).fill(0));
  };

  return (
    <div className="min-h-screen w-full bg-gray-900 flex flex-col gap-4 p-8 text-white items-center">
      <h1 
        className="text-4xl font-bold mb-4"
        style={{ 
          color: '#FF9E00',
          textShadow: `
            0 0 20px #FF9E00AA,
            0 0 40px #FF9E0088,
            0 0 60px #FF9E0044,
             2px 2px 2px rgba(0, 0, 0, 0.5)
          `
        }}
      >
        Multi-Layer Perceptron Visualization
      </h1>
   
      {errorMessage && (
        <div className="px-4 py-2 bg-red-900/50 border border-red-500 rounded mb-4 text-red-200">
          {errorMessage}
        </div>
      )}
      <div className="flex gap-4 items-start justify-center">
        <canvas
          ref={canvasRef}
          width={CANVAS_SIZE}
          height={CANVAS_SIZE}
          className="border border-gray-600 cursor-crosshair"
          onMouseDown={(e) => {
            setIsDrawing(true);
            handleMouseMove(e);
          }}
          onMouseMove={handleMouseMove}
          onMouseUp={() => setIsDrawing(false)}
          onMouseLeave={() => setIsDrawing(false)}
        />
        <div className="flex flex-col gap-2">
 
          <button
            onClick={clearCanvas}
            className="mb-1 px-4 py-2 bg-gray-800 text-[#00E5FF] border border-[#00E5FF] rounded hover:bg-[#00E5FF22] transition-colors font-medium"
          >
            Clear
          </button>
     
          <input
            type="file"
            accept=".json"
            onChange={handleWeightUpload}
            className="hidden"
            id="weight-upload"
          />
          <label
            htmlFor="weight-upload"
            className="px-4 py-2 bg-gray-800 text-[#FF9E00] border border-[#FF9E00] rounded hover:bg-[#FF9E0022] transition-colors font-medium cursor-pointer text-center"
          >
            Load Weights
          </label>
          <style jsx>{`
          select.custom-dropdown {
            text-align: center; /* Center the text */
            text-align-last: center; /* Center the selected text */
          }          
          select.custom-dropdown option {
            background-color: #000000; /* Change this to your desired color */
            color: #FFFFFFE0; /* Change this to your desired text color */
            text-align: center;
          }
        `}</style>

          <select
            value={selectedModel}
            onChange={(e) => {
              setSelectedModel(e.target.value);
              handleModelSelect(e.target.value);
            }}
            className="px-4 py-2 bg-gray-800  text-[#FFFFFF] border border-[#FFFFFF80] hover:bg-[#FFFFFF22] rounded custom-dropdown"
          >
            {availableModels.map(model => (
              <option key={model.name} value={model.name}>
                {model.name}
              </option>
            ))}
          </select>


          {networkConfig.description && (

            <div className="text-sm text-gray-300 min-w-[250px] mt-2 px-2 text-center">
            {networkConfig.description.split('\n').map((line, index) => (
              <Fragment key={index}>
                {line}
                <br />
              </Fragment>
            ))}
            </div>

          )}
         <div className="text-sm text-[#FF009E] min-w-[250px] mt-1 px-2 text-center">
          <a 
            href="https://github.com/cpldcpu/neural-network-visualizer" 
            target="_blank" 
            rel="noopener noreferrer" 
            className="flex items-center justify-center mt-1"
          >
            <img 
              src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" 
              alt="GitHub Logo" 
              className="w-4 h-4 mr-1"
            /> Link to Repository </a>
            </div>
        </div>
      </div>
      
      <NetworkViz 
        activations={pixelData} 
        config={networkConfig}
        centerInput={centerInput}
        setCenterInput={setCenterInput}
      />
    </div>
  );
};

export default DrawingCanvas;