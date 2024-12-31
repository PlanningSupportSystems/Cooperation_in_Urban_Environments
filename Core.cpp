#include "gdal_priv.h"
#include "ogrsf_frmts.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <filesystem> // For directory operations
#include <fstream>
#include <QDataStream>
#include <random>
#include <unordered_map> // Include this for unordered_map
#include <queue>
#include <QtConcurrent>
#include <QFuture>
#include <set>
#include <algorithm>
#include <QAtomicInt>
#include <cstdlib>
#include <ctime>
#include <unordered_set>
#include <cmath>
#include <functional>  // For std::greater
#include <QImage>
#include <QPixmap>
#include <QObject>
#include <QThread>
#include <QLinearGradient>
#include <QColor>
#include <QDateTime>
#include <Qprocess>
//#include "AStarCuda.cuh"
#include "SimpleNode.h"
#include <opencv2/opencv.hpp>
#include <QMessageBox>
#include "gdal.h"
#include "ogr_geometry.h"
#include "Core.h"
//#include <sstream>
#include "gnuplot-iostream.h"

//std::vector<PolygonData> polygonsLocalCore;

// Define the PairHash structure
size_t PairHash::operator()(const std::pair<int, int>& p) const {
    auto h1 = std::hash<int>{}(p.first);
    auto h2 = std::hash<int>{}(p.second);
    return h1 ^ h2;
}

size_t PairHash::operator()(const std::tuple<int, int, int, int>& t) const {
    auto h1 = std::hash<int>{}(std::get<0>(t));
    auto h2 = std::hash<int>{}(std::get<1>(t));
    auto h3 = std::hash<int>{}(std::get<2>(t));
    auto h4 = std::hash<int>{}(std::get<3>(t));
    return h1 ^ h2 ^ h3 ^ h4;
}


// Assignment operator for deep copy
PathInfo& PathInfo::operator=(const PathInfo& other) {
    if (this != &other) {
        cost = other.cost;
        path = other.path;  // Deep copy the vector
        Length = other.Length;
        totalVisitedNodes = other.totalVisitedNodes;
        source = other.source;
        destination = other.destination;
    }
    return *this;
}

// Define the ComparePathNode structure

bool ComparePathNode::operator()(const PathNode& left, const PathNode& right) const {
    return left.costSoFar + left.heuristic > right.costSoFar + right.heuristic;
}


using pathMap = std::unordered_map<std::tuple<int, int, int, int>, PathInfo, PairHash>;

void ABM::calculateShortestPathsCUDA(const std::vector<std::vector<Node>>& nodeGrid, int startRow,int startCol,int destRow,int destCol) {
    int numRows = nodeGrid.size();
    int numCols = nodeGrid[0].size();
    SimpleNode* simpleNodes = new SimpleNode[numRows * numCols];

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            int idx = i * numCols + j;
            simpleNodes[idx].row = nodeGrid[i][j].row;
            simpleNodes[idx].col = nodeGrid[i][j].col;
            simpleNodes[idx].trait3 = nodeGrid[i][j].trait3;
            simpleNodes[idx].isPath = nodeGrid[i][j].isPath;
            simpleNodes[idx].accessPointID = nodeGrid[i][j].accessPointID;
        }
    }

    // Call CUDA function
    //calculateShortestPathsAStarCUDA2(simpleNodes, numRows, numCols, startRow,  startCol, destRow, destCol);

    // Handle results here
    delete[] simpleNodes;
}

// Define a function to calculate the color of each node in the gridHeatPath
void ABM::calculateNodeColors(GridHeatPath& gridHeatPath) {
    // Find the maximum score in the grid
    int maxScore = 0;
    for (const auto& row : gridHeatPath.heatnodes) {
        for (const auto& node : row) {
            maxScore = std::max(maxScore, node.score);
        }
    }

    // Iterate over each node in the grid
    for (auto& row : gridHeatPath.heatnodes) {
        for (auto& node : row) {
            if (node.score > 0) {
                // Apply logarithmic normalization to flatten outlier values
                double normalizedScore = std::log10(static_cast<double>(node.score) + 1) / std::log10(maxScore + 1);

                // Calculate the RGB values based on the normalized score
                int r, g, b;
                if (normalizedScore < 0.25) {
                    r = 0;
                    g = static_cast<int>(255 * (4 * normalizedScore));
                    b = 255;
                }
                else if (normalizedScore < 0.5) {
                    r = 0;
                    g = 255;
                    b = static_cast<int>(255 * (1 - 4 * (normalizedScore - 0.25)));
                }
                else if (normalizedScore < 0.75) {
                    r = static_cast<int>(255 * (4 * (normalizedScore - 0.5)));
                    g = 255;
                    b = 0;
                }
                else {
                    r = 255;
                    g = static_cast<int>(255 * (1 - 4 * (normalizedScore - 0.75)));
                    b = 0;
                }

                // Set the color of the node
                node.color = QColor(r, g, b);
            }
        }
    }
}

// Function to update the corresponding GridNodeHeatPath score based on the path
void ABM::updateGridHeatPath(int sourceRow, int sourceCol, int destinationRow, int destinationCol, const pathMap& pathmap, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, int alphaSteps, int betaSteps) {
    // Search for the PathInfo corresponding to the source and destination in the pathmap
    auto it = pathmap.find(std::make_tuple(sourceRow, sourceCol, destinationRow, destinationCol));

    // If the normal path is not found, search for the inverse path
    if (it == pathmap.end()) {
        it = pathmap.find(std::make_tuple(destinationRow, destinationCol, sourceRow, sourceCol));
    }

    // If a path (normal or inverse) is found, update the gridHeatPath
    if (it != pathmap.end()) {
        const PathInfo& pathInfo = it->second;

        // Iterate over the path and update the corresponding GridNodeHeatPath score
        for (const auto& nodePair : pathInfo.path) {
            int row = nodePair.first;
            int col = nodePair.second;

            // Check if the row and col are within the grid bounds
            if (row >= 0 && row < gridHeatPath.heatnodes.size() && col >= 0 && col < gridHeatPath.heatnodes[0].size()) {
                gridHeatPath.heatnodes[row][col].score++;
            }
        }
    }

    // Optionally update gridHeatPathVector if needed
    // gridHeatPathVector[alphaSteps][betaSteps] = gridHeatPath;
}


// Function to convert polygon coordinates to grid
GridHeatPath ABM::convertPolygonsToGrid(const std::vector<PolygonData>& polygons, int gridWidth, int gridHeight) {
    // Find the bounding box of all polygons
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::min();
    double maxY = std::numeric_limits<double>::min();

    for (const auto& polygon : polygons) {
        for (const auto& point : polygon.points) {
            minX = std::min(minX, point.x());
            minY = std::min(minY, point.y());
            maxX = std::max(maxX, point.x());
            maxY = std::max(maxY, point.y());
        }
    }

    // Calculate the size of each grid cell
    double cellWidth = (maxX - minX) / gridWidth;
    double cellHeight = (maxY - minY) / gridHeight;

    // Initialize the grid
    GridHeatPath gridheatpath;
    gridheatpath.heatnodes.resize(gridHeight, std::vector<GridNodeHeatPath>(gridWidth));

    // Iterate through each grid cell to calculate row, col, and coordinates
    for (int row = 0; row < gridHeight; ++row) {
        for (int col = 0; col < gridWidth; ++col) {
            // Calculate the x, y coordinates of the grid cell
            double x = minX + col * cellWidth + cellWidth / 2.0;
            double y = minY + row * cellHeight + cellHeight / 2.0;

            // Map the row, col, and coordinates to the grid node
            gridheatpath.heatnodes[row][col] = { row, col,0, QColor(), QPointF(x, y) };
        }
    }

    return gridheatpath;
}

std::string ABM::extractFilename(std::string fullPath) {
    // Convert the const char* to std::string
    std::string fullPathStr = fullPath;

    // Find the position of the last '\' character
    size_t lastSlashPos = fullPathStr.find_last_of("\\");
    if (lastSlashPos == std::string::npos) {
        // No '\' character found, return empty string or fullPath as per requirement
        return "";
    }

    // Find the position of the last '.' character
    size_t lastDotPos = fullPathStr.find_last_of(".");
    if (lastDotPos == std::string::npos || lastDotPos < lastSlashPos) {
        // No '.' character found after the last '\', or '.' character is before '\', return empty string
        return "";
    }

    // Extract the substring between the last '\' and the last '.'
    return fullPathStr.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);
}

double ABM::calculateAverageTrait2Value(const Agent& agent) {
    double sum = 0.0;
    for (double value : agent.trait2Values) {
        sum += value;
    }
    return sum / agent.trait2Values.size();
}

int ABM::getHighestVisitedNodes(const pathMap& pathMap) {
    int highestVisitedNodes = 0;

    for (const auto& entry : pathMap) {
        const auto& pathInfo = entry.second;
        int totalVisitedNodes = pathInfo.path.size();
        highestVisitedNodes = std::max(highestVisitedNodes, totalVisitedNodes);
    }
    std::cout << "Highest Visited Nodes" << highestVisitedNodes << std::endl;
    emit logMessage("Highest Visited Nodes " + QString::number(highestVisitedNodes));
    return highestVisitedNodes;
}

double ABM::calculateMedianVisitedNodes(const pathMap& pathMap) {
    std::vector<int> visitedNodes;

    // Extract all totalVisitedNodes values from the pathMap
    for (const auto& entry : pathMap) {
        const auto& pathInfo = entry.second;
        visitedNodes.push_back(pathInfo.totalVisitedNodes);
    }

    // Sort the extracted values
    std::sort(visitedNodes.begin(), visitedNodes.end());

    int numValues = visitedNodes.size();
    if (numValues == 0) {
        return 0;  // No values, return 0
    }

    // Calculate the median
    if (numValues % 2 == 0) {
        // Even number of values
        return (visitedNodes[numValues / 2 - 1] + visitedNodes[numValues / 2]) / 2.0;
    }
    else {
        // Odd number of values
        return visitedNodes[numValues / 2];
    }
}

// Function to find the minimum and maximum trait values in the nodeGrid
std::pair<double, double> ABM::findTraitBounds(const std::vector<std::vector<Node>>& nodeGrid) {
    double minTrait = std::numeric_limits<double>::max();
    double maxTrait = std::numeric_limits<double>::lowest();

    for (const auto& row : nodeGrid) {
        for (const auto& node : row) {
            minTrait = std::min(minTrait, node.trait3);
            maxTrait = std::max(maxTrait, node.trait3);
        }
    }

    return { minTrait, maxTrait };
}

// Function to generate a random value between the upper and lower bounds
double ABM::generateRandomValue(double lowerBound, double upperBound) {
    if (lowerBound < 1) {
        lowerBound = 1;
    }
    std::random_device rd; // Obtain a random number from hardware
    std::mt19937 eng(rd()); // Seed the random number generator
    std::uniform_real_distribution<double> distr(lowerBound, upperBound); // Define the range

    return distr(eng); // Generate and return a random value
}

// Function to calculate Shannon entropy of agent traits
double ABM::calculateAgentShannonEntropy(const std::vector<Agent>& agents) {
    std::unordered_map<double, int> traitFrequency;
    int totalAgents = agents.size();

    // Step 1: Calculate frequency of each unique trait value
    for (const auto& agent : agents) {
        traitFrequency[std::round(agent.trait2 * 10) / 10]++;
    }

    // Step 2: Calculate probability of each unique trait value
    std::unordered_map<double, double> traitProbability;
    for (const auto& pair : traitFrequency) {
        double probability = static_cast<double>(pair.second) / totalAgents;
        traitProbability[pair.first] = probability;
    }

    // Step 3: Calculate Shannon entropy
    double entropy = 0.0;
    for (const auto& pair : traitProbability) {
        double probability = pair.second;
        entropy -= probability * std::log2(probability);
        //std::cout << "Entropia foi calculada " << entropy << std::endl;
    }
    //std::cout << "Entropia dos agentes " << entropy << std::endl;
    return entropy;
}

// Function to calculate Shannon entropy
double ABM::calculateShannonEntropyGRID(const std::vector<std::vector<Node>>& nodeGrid) {
    std::unordered_map<double, int> traitFrequency;
    int totalNodes = 0;

    // Step 1: Calculate frequency of each unique trait value
    for (const auto& row : nodeGrid) {
        for (const auto& node : row) {
            traitFrequency[std::round(node.trait3 * 10) / 10]++;
            totalNodes++;
        }
    }

    // Step 2: Calculate probability of each unique trait value
    std::unordered_map<double, double> traitProbability;
    for (const auto& pair : traitFrequency) {
        double probability = static_cast<double>(pair.second) / totalNodes;
        traitProbability[pair.first] = probability;
    }

    // Step 3: Calculate Shannon entropy
    double entropy = 0.0;
    for (const auto& pair : traitProbability) {
        double probability = pair.second;
        entropy -= probability * std::log2(probability);
    }
    //std::cout << "Entropia dos lugares " << entropy << std::endl;
    return entropy;
}

double ABM::perpendicularDistance(const std::pair<int, int>& point, const std::pair<int, int>& lineStart, const std::pair<int, int>& lineEnd) {
    double x1 = lineStart.first;
    double y1 = lineStart.second;
    double x2 = lineEnd.first;
    double y2 = lineEnd.second;
    double x0 = point.first;
    double y0 = point.second;

    double dx = x2 - x1;
    double dy = y2 - y1;

    if (dx == 0 && dy == 0) {
        // Line segment is a point
        return std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
    }

    double u = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx * dx + dy * dy);
    double intersectionX, intersectionY;

    if (u < 0) {
        intersectionX = x1;
        intersectionY = y1;
    }
    else if (u > 1) {
        intersectionX = x2;
        intersectionY = y2;
    }
    else {
        intersectionX = x1 + u * dx;
        intersectionY = y1 + u * dy;
    }

    return std::sqrt((intersectionX - x0) * (intersectionX - x0) + (intersectionY - y0) * (intersectionY - y0));
}


// Function to save paths from pathMap to a CSV file in QGIS format with EPSG conversion
void ABM::savePathsToCSV(const std::string& filename, const pathMap& paths) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }

    // Write CSV header
    file << "path_id;cost;node_id;x;y" << std::endl;

    // Iterate over paths in pathMap
    int pathId = 0;
    for (const auto& entry : paths) {
        const PathInfo& pathInfo = entry.second;

        // Write path information
        for (size_t nodeIndex = 0; nodeIndex < pathInfo.path.size(); ++nodeIndex) {
            const std::pair<int, int>& node = pathInfo.path[nodeIndex];

            // Assuming you have node coordinates (x, y)
            int x = node.first;
            int y = node.second;

            // Write CSV row
            file << pathId << ";" << pathInfo.cost << ";" << nodeIndex << ";" << x << ";" << y << std::endl;
        }

        ++pathId;
    }

    file.close();
}

void ABM::exportImageFromNodeGrid(const std::vector<std::vector<Node>>& nodeGrid) {
    // Define the size of the image based on the number of rows and columns in nodeGrid
    int numRows = static_cast<int>(nodeGrid.size());
    int numCols = numRows > 0 ? static_cast<int>(nodeGrid[0].size()) : 0;

    // Create a 3-channel image (BGR) with white background
    cv::Mat image(numRows, numCols, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat image2(numRows, numCols, CV_8UC3, cv::Scalar(255, 255, 255));

    // Create a color map for unique trait3 values
    std::unordered_map<int, cv::Scalar> colorMap;
    std::unordered_map<int, cv::Scalar> colorMap2;

    // Assign a unique color for each unique trait3 value
    int colorIndex = 0;
    int colorIndex2 = 0;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            int trait3Value = nodeGrid[i][j].trait3;
            int IDValue = nodeGrid[i][j].id;
            int acessPoint = nodeGrid[i][j].accessPointID;

            if (colorMap.find(trait3Value) == colorMap.end()) {
                // Generate a unique color for the trait3 value
                colorMap[trait3Value] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
                ++colorIndex;
            }
            if (colorMap2.find(IDValue) == colorMap2.end()) {
                // Generate a unique color for the trait3 value
                colorMap2[IDValue] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
                ++colorIndex2;
            }

            // Set the pixel color in the image based on the trait3 value
            if (trait3Value == 0) {
                image.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);  // White color
                image2.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);  // White color
            }
            else {
                if (acessPoint == 0) {
                    image.at<cv::Vec3b>(i, j) = cv::Vec3b(colorMap[trait3Value][0], colorMap[trait3Value][1], colorMap[trait3Value][2]);
                    image2.at<cv::Vec3b>(i, j) = cv::Vec3b(colorMap2[IDValue][0], colorMap2[IDValue][1], colorMap2[IDValue][2]);
                }
                else {
                    image.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);  // RED color
                    image2.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);  // RED color
                }
            }
        }
    }

    // Convert the OpenCV Mat images to QImage
    QImage imageVIEW2(reinterpret_cast<uchar*>(image.data), image.cols, image.rows, image.step, QImage::Format_RGB888);
    QImage imageVIEW(reinterpret_cast<uchar*>(image2.data), image2.cols, image2.rows, image2.step, QImage::Format_RGB888);

    //emit ImageProduced(imageVIEW); // Return the QImage object
    //emit ImageProduced2(imageVIEW2); // Return the QImage object
}

void ABM::createImageFromNodeGrid(const std::vector<std::vector<Node>>& nodeGrid, const std::string& outputFilename) {
    // Define the size of the image based on the number of rows and columns in nodeGrid
    int numRows = static_cast<int>(nodeGrid.size());
    int numCols = numRows > 0 ? static_cast<int>(nodeGrid[0].size()) : 0;

    // Create a 3-channel image (BGR) with white background
    cv::Mat image(numRows, numCols, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat image2(numRows, numCols, CV_8UC3, cv::Scalar(255, 255, 255));

    // Create a color map for unique trait3 values
    std::unordered_map<int, cv::Scalar> colorMap;
    std::unordered_map<int, cv::Scalar> colorMap2;

    // Assign a unique color for each unique trait3 value
    int colorIndex = 0;
    int colorIndex2 = 0;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            int trait3Value = nodeGrid[i][j].trait3;
            int IDValue = nodeGrid[i][j].id;
            int acessPoint = nodeGrid[i][j].accessPointID;
            //std::cout << "accessPointID " << acessPoint << std::endl;
            if (colorMap.find(trait3Value) == colorMap.end()) {
                // Generate a unique color for the trait3 value
                colorMap[trait3Value] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
                ++colorIndex;
            }
            if (colorMap2.find(IDValue) == colorMap2.end()) {
                // Generate a unique color for the trait3 value
                colorMap2[IDValue] = cv::Scalar(rand() % 256, rand() % 256, rand() % 256);
                ++colorIndex2;
            }
            // Set the pixel color in the image based on the trait3 value
            if (trait3Value == 0) {
                image.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);  // White color
                image2.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);  // White color
            }
            else {
                if (acessPoint == 0) {
                    image.at<cv::Vec3b>(i, j) = cv::Vec3b(colorMap[trait3Value][0], colorMap[trait3Value][1], colorMap[trait3Value][2]);
                    image2.at<cv::Vec3b>(i, j) = cv::Vec3b(colorMap2[IDValue][0], colorMap2[IDValue][1], colorMap2[IDValue][2]);
                }
                else {
                    image.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);  // RED color
                    image2.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 0, 0);  // RED color
                    //std::cout << "E ponto de acesso local ( " << i << ", " << j << ")" << std::endl;
                }
            }


        }
    }

    // Save the image to the specified output file
    cv::imwrite(outputFilename + "_USO.png", image);
    cv::imwrite(outputFilename + "_IDs.png", image2);

    emit ImageProduced(QString::fromStdString(outputFilename));
}

std::vector<Agent> ABM::initializeAgents(const std::vector<std::vector<Node>>& nodeGrid, int numAgents, double minTrait, double maxTrait) {
    std::vector<Agent> agents;

    // Map to associate access point IDs with their corresponding nodes
    std::vector<std::vector<AgentNode>> accessPointMap(nodeGrid.size(), std::vector<AgentNode>(nodeGrid[0].size()));

    // Get all valid access points and populate the accessPointMap
    for (int i = 0; i < nodeGrid.size(); ++i) {
        for (int j = 0; j < nodeGrid[i].size(); ++j) {
            const Node& node = nodeGrid[i][j];

            if (node.accessPointID > 0 && node.trait3 != -1) {
                AgentNode agentNode;
                agentNode.id = node.id;
                agentNode.trait2 = node.trait3;
                agentNode.accessPointID = node.accessPointID;
                agentNode.row = i;  // Use the current loop indices
                agentNode.col = j;

                accessPointMap[i][j] = agentNode;
            }
        }
    }

    // Seed for randomization (call it once at the beginning)
    std::srand(static_cast<unsigned>(std::time(0)));

    // Shuffle the valid access points using std::shuffle
    std::vector<AgentNode> shuffledAccessPoints;
    for (const auto& row : accessPointMap) {
        for (const AgentNode& agentNode : row) {
            if (agentNode.accessPointID > 0) {
                shuffledAccessPoints.push_back(agentNode);
            }
        }
    }
    std::shuffle(shuffledAccessPoints.begin(), shuffledAccessPoints.end(), std::default_random_engine(std::rand()));

    emit logMessage("quantidade de locais para criar agentes: " + QString::number(shuffledAccessPoints.size()));

    // Distribute trait3 values uniformly among the agents
    if (DistribuicaoAgentesLocal == 0) {
        // Filter out -1 and 0 from the trait values
        std::vector<int> traitValues;
        for (int i = static_cast<int>(1); i <= static_cast<int>(MaxtraitsAgentsLocal); ++i) {
            if (i != -1 && i != 0) {
                traitValues.push_back(i);
            }
        }

        int numTraits = static_cast<int>(traitValues.size());
        int agentsPerTrait = numAgents / numTraits;
        int remainderAgents = numAgents % numTraits;

        // Distribute trait values to agents
        int traitIndex = 0;
        int traitCounter = 0;  // Count how many agents have been assigned the current trait

        for (int i = 0; i < numAgents && i < shuffledAccessPoints.size(); ++i) {
            const AgentNode& agentNode = shuffledAccessPoints[i];

            // Assign the current trait value
            int assignedTrait = traitValues[traitIndex];
            Agent agent;
            agent.trait2 = assignedTrait;
            agent.trait2Values.push_back(assignedTrait);
            agent.latentTrait = assignedTrait;
            agent.currentAccessPointID = agentNode.id;
            agent.AccessPointIDValues.push_back(agentNode.id);
            agent.row = agentNode.row;
            agent.col = agentNode.col;
            agent.initrow = agentNode.row;
            agent.initcol = agentNode.col;
            agent.name = std::to_string(i+1);

            // Log the agent creation
            //std::cout << "Agent " << i + 1 << " created with Trait " << assignedTrait << " at Access Point (" << agentNode.id << ") at position (" << agentNode.row << ", " << agentNode.col << ")." << std::endl;
            emit logMessage("Agent " + QString::number(i+1) + " created with Trait " + QString::number(assignedTrait) + " at Access Point (" + QString::number(agentNode.id) + ") at position (" + QString::number(agentNode.row) + ", " + QString::number(agentNode.col) + ").");

            agents.push_back(agent);

            // Increment the traitCounter and check if we need to move to the next trait
            traitCounter++;
            if (traitCounter >= agentsPerTrait + (remainderAgents > 0 ? 1 : 0)) {
                traitIndex = (traitIndex + 1) % numTraits;
                traitCounter = 0;
                if (remainderAgents > 0) {
                    remainderAgents--;
                }
            }
        }
    }
    else if (DistribuicaoAgentesLocal == 2) {

        for (int i = 0; i < numAgents && i < shuffledAccessPoints.size(); ++i) {
            const AgentNode& agentNode = shuffledAccessPoints[i];
            // Find the trait corresponding to the agent's ID, which is (i + 1) in agentTraitsLocal

            //emit logMessage("Contents of agentTraitsLocal:");
            //for (const auto& pair : agentTraitsLocal) {
            //    emit logMessage("ID: " + QString::number(pair.first) + ", Trait: " + QString::number(pair.second));
            //}

            int agentId = i + 1;
            auto it = std::find_if(agentTraitsLocal.begin(), agentTraitsLocal.end(),
                [agentId](const std::pair<int, int>& p) {
                    return p.first == agentId;
                });

            double randomValue = 0.0;
            if (it != agentTraitsLocal.end()) {
                randomValue = static_cast<double>(it->second); // Set the corresponding trait value
            }

            // Create an agent at the selected node
            Agent agent;
            agent.trait2 = randomValue;
            agent.trait2Values.push_back(randomValue);
            agent.latentTrait = randomValue;
            agent.currentAccessPointID = agentNode.id;
            agent.AccessPointIDValues.push_back(agentNode.id);
            agent.row = agentNode.row;
            agent.col = agentNode.col;
            agent.initrow = agentNode.row;
            agent.initcol = agentNode.col;
            agent.name = std::to_string(i + 1);

            // Log the agent creation
            emit logMessage("Agent " + QString::number(i + 1) + " created with Trait " + QString::number(agent.trait2) + " at Access Point (" + QString::number(agentNode.id) + ") at position (" + QString::number(agentNode.row) + ", " + QString::number(agentNode.col) + ").");

            agents.push_back(agent);
        }
    }

    else {
        // Original logic or other distributions go here
        for (int i = 0; i < numAgents && i < shuffledAccessPoints.size(); ++i) {
            const AgentNode& agentNode = shuffledAccessPoints[i];
            double randomValue = generateRandomValue(1, MaxtraitsAgentsLocal);

            // Create an agent at the selected node
            Agent agent;
            agent.trait2 = randomValue;
            agent.trait2Values.push_back(randomValue);
            agent.latentTrait = randomValue;
            agent.currentAccessPointID = agentNode.id;
            agent.AccessPointIDValues.push_back(agentNode.id);
            agent.row = agentNode.row;
            agent.col = agentNode.col;
            agent.initrow = agentNode.row;
            agent.initcol = agentNode.col;
            agent.name = std::to_string(i+1);

            // Log the agent creation
            //std::cout << "Agent " << i + 1 << " created at Access Point (" << agentNode.id << ") at position (" << agentNode.row << ", " << agentNode.col << ")." << std::endl;
            emit logMessage("Agent " + QString::number(i+1) + " created with Trait " + QString::number(agent.trait2) + " at Access Point (" + QString::number(agentNode.id) + ") at position (" + QString::number(agentNode.row) + ", " + QString::number(agentNode.col) + ").");
            //emit logMessage("Agent " + QString::number(i + 1) + " created at Access Point (" + QString::number(agentNode.id) + ") at position (" + QString::number(agentNode.row) + ", " + QString::number(agentNode.col) + ").");

            agents.push_back(agent);
        }
    }

    return agents;
}



double ABM::calculateScore(const Agent& agent, int row, int col, const Node& targetNode, const pathMap& pathMap, int currentRow, int currentCol, double alpha, double beta, double Maiorcaminho, double maxTrait) {
    double traitDifference = std::abs(targetNode.trait3 - agent.trait2);
    double traitScore = traitDifference / maxTrait;
    //std::cout << "traitDifference" << traitDifference << std::endl;
    //std::cout << "traitScore" << traitScore << std::endl;
    //std::cout << "Maiorcaminho" << Maiorcaminho << std::endl;

    // Set individual values
    std::tuple<int, int, int, int> key = { currentRow, currentCol, row, col };  // Swap values in the key
    std::tuple<int, int, int, int> keyReverse = { row, col, currentRow, currentCol };


    //double distance = 0;
    //std::cout << "tentando acessar: (" << currentRow << ", " << currentCol << ", " << row << ", " << col << ")" << std::endl;
    // Retrieve PathInfo for the specified source and destination
    //std::cout << "chegou no calculo pelo menos" << std::endl;
    for (const auto& entry : pathMap) {
        const auto& key = entry.first;
        //std::cout << "Key: (" << std::get<0>(key) << ", " << std::get<1>(key) << ", " << std::get<2>(key) << ", " << std::get<3>(key) << ")" << std::endl;
    }

    if (pathMap.count(key) > 0) {
        const auto& pathInfo = pathMap.at(key);

        // Access the totalVisitedNodes from PathInfo
        double distance = pathInfo.path.size();
        //std::cout << "distancia em steps" << distance << std::endl;
        //std::cout << "caminho atual" << pathInfo.totalVisitedNodes << std::endl;
        //std::cout << "distance" << distance << std::endl;
        //std::cout << "Key: (" << std::get<0>(key) << ", " << std::get<1>(key) << ", " << std::get<2>(key) << ", " << std::get<3>(key) << ")" << std::endl;
        double distanceScore = distance / Maiorcaminho;
        //std::cout << "distanceScore" << distanceScore << std::endl;
        // Adjust scaling factors to achieve balanced contribution
        //double adjustedAlpha = alpha * distance / traitDifference;
        //double adjustedBeta = beta * traitDifference / distance;

        //emit logMessage("beta = " + QString::number(beta));

        double finalScore = (traitScore - (alpha * traitScore)) + (distanceScore - ( beta * distanceScore));
        //std::cout << "Final Score: " << finalScore << std::endl;
        //std::cout << "traitScore : " << traitScore  << std::endl;
        //std::cout << "distanceScore : " << distanceScore  << std::endl;
        //std::cout << "Final Score: " << finalScore << std::endl;
        //return finalScore;

        //std::cout << "Final Score: " << finalScore << std::endl;

        if (finalScore == 0.0) {
            std::cout << "Final Score: " << finalScore << std::endl;
        }

        return finalScore;
    }
    else if (pathMap.count(keyReverse) > 0) {
        const auto& pathInfo = pathMap.at(keyReverse);

        // Access the totalVisitedNodes from PathInfo
        double distance = pathInfo.path.size();
        //std::cout << "caminho atual" << pathInfo.totalVisitedNodes << std::endl;
        //std::cout << "distance" << distance << std::endl;
        //std::cout << "distancia em steps" << distance << std::endl;
        //std::cout << "KeyReverse: (" << std::get<0>(keyReverse) << ", " << std::get<1>(keyReverse) << ", " << std::get<2>(keyReverse) << ", " << std::get<3>(keyReverse) << ")" << std::endl;
        double distanceScore = 1.0 - distance / Maiorcaminho;
        //std::cout << "distanceScore" << distanceScore << std::endl;
        //Adjust scaling factors to achieve balanced contribution
        //double adjustedAlpha = alpha * distance / traitDifference;
        //double adjustedBeta = beta * traitDifference / distance;

        double finalScore = (traitScore - (alpha * traitScore)) + (distanceScore - (beta * distanceScore));
        //std::cout << "Final Score: " << finalScore << std::endl;
        //std::cout << "traitScore : " << traitScore << std::endl;
        //std::cout << "distanceScore : " << distanceScore << std::endl;
        //std::cout << "Final Score: " << finalScore << std::endl;
        //return finalScore;

        //std::cout << "Final Score: " << finalScore << std::endl;
        if (finalScore == 0.0) {
            std::cout << "Final Score: " << finalScore << std::endl;
        }

        return finalScore;
    }
    else {
        std::cout << "nao existe caminho entre os pontos (" << currentRow << "," << currentCol << ") e (" << row << ", " << col << ")" << std::endl;
        return std::numeric_limits<double>::infinity();
    }
    // Debug prints
    //std::cout << "Trait Difference: " << traitDifference << std::endl;
    //std::cout << "Trait Score: " << traitScore << std::endl;
    //std::cout << "Distance: " << distance << std::endl;
    //std::cout << "Distance Score: " << distanceScore << std::endl;


}

std::tuple<int, int, int> ABM::calculateNextAccessPoint(const Agent& agent, const std::vector<std::vector<Node>>& nodeGrid, const pathMap& pathMap, double alpha, double beta, double Maiorcaminho, double maxTrait, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, int alphaSteps, int betaSteps) {
    
    int currentRow = 0;
    int currentCol = 0;

    if (VoltarPraCasaLocal == 0) {
        currentRow = agent.initrow;
        currentCol = agent.initcol;
    }
    else {
        currentRow = agent.row;
        currentCol = agent.col;
    }

    int numRows = static_cast<int>(nodeGrid.size());
    int numCols = static_cast<int>(nodeGrid[0].size());

    std::vector<std::tuple<double, int, int, int>> bestScoresWithCoords;

    double bestScore = std::numeric_limits<double>::infinity();
    int bestAccessPointID = -1;
    int bestrow = currentRow;
    int bestcol = currentCol;

    /*
    for (int row = 0; row < numRows; ++row) {
        for (int col = 0; col < numCols; ++col) {
            // Skip the current access point of the agent
            if (!(nodeGrid[row][col].accessPointID > 0) || (row == currentRow && col == currentCol) || (row == agent.row && col == agent.col)) {
                //std::cout << "Pulou o proprio node em " << std::endl;
                continue;
            }
            const Node& targetNode = nodeGrid[row][col];
            double score = calculateScore(agent, row, col, targetNode, pathMap, currentRow, currentCol, alpha, beta, Maiorcaminho, maxTrait);

            // Print debugging information
            //emit logMessage("Agent at (" + QString::number(currentRow) + ", " + QString::number(currentCol) + ") moving to access point {" + QString::number(nodeGrid[row][col].id) + "} at (" + QString::number(row) + ", " + QString::number(row) + ") with score : " + QString::number(score));
            //std::cout << "Agent at (" << currentRow << ", " << currentCol << ") moving to access point at (" << row << ", " << col << ") with score: " << score << std::endl;

            //if (score < bestScore) {
            //    bestScore = score;
                //std::cout << "\nNew best score: " << bestScore << std::endl;
            //    bestAccessPointID = targetNode.id;
            //    bestrow = row;
            //    bestcol = col;
            //}
            if (score != std::numeric_limits<double>::infinity()) {

                bestScoresWithCoords.push_back(std::make_tuple(score, row, col, targetNode.id)); // Store the score along with its row and column coordinates
            }
            
        }
    }
    */

    for (const auto& entry : accessPointMap) {
        int accessPointID = entry.first;
        int row = entry.second.first;
        int col = entry.second.second;

        // Skip the current access point of the agent
        if ((row == currentRow && col == currentCol) || (row == agent.row && col == agent.col)) {
            continue;
        }

        const Node& targetNode = nodeGrid[row][col];
        double score = calculateScore(agent, row, col, targetNode, pathMap, currentRow, currentCol, alpha, beta, Maiorcaminho, maxTrait);

        if (score != std::numeric_limits<double>::infinity()) {
            bestScoresWithCoords.push_back(std::make_tuple(score, row, col, targetNode.id)); // Store the score along with its row and column coordinates
        }
    }

    // Remove entries with infinity scores
    //bestScoresWithCoords.erase(
    //    std::remove_if(bestScoresWithCoords.begin(), bestScoresWithCoords.end(),
    //        [](const std::tuple<double, int, int, int>& entry) {
    //            return std::get<0>(entry) == std::numeric_limits<double>::infinity();
    //        }),
    //    bestScoresWithCoords.end()
    //);

    // Sort the vector to get the best scores ordered by value
    std::sort(bestScoresWithCoords.begin(), bestScoresWithCoords.end());
    int index;

    if (metodoEscolhaLocal == 0) {

        // Exponential decay factor
        double lambda = 0.1; // Adjust this value to control the skew
        double totalExponentialSum = 0.0;

        std::vector<double> exponentialWeights;
        for (int i = 0; i < bestScoresWithCoords.size(); ++i) {
            double weight = std::exp(-lambda * i);
            exponentialWeights.push_back(weight);
            totalExponentialSum += weight;
        }

        // Normalize the probabilities
        std::vector<double> probabilities;
        for (double weight : exponentialWeights) {
            probabilities.push_back(weight / totalExponentialSum);
        }

        // Randomly select one of the best scores with its respective row and column coordinates based on probabilities
        std::default_random_engine generator;
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
        index = distribution(generator);
        //emit logMessage("Metodo exponencial");
    }
    else if(metodoEscolhaLocal == 1) {
        // Calculate the total sum of indices
        int totalIndexSum = bestScoresWithCoords.size() * (bestScoresWithCoords.size() + 1) / 2;

        // Calculate probabilities inversely proportional to the index
        std::vector<double> probabilities;
        for (int i = 0; i < bestScoresWithCoords.size(); ++i) {
            double probability = static_cast<double>(bestScoresWithCoords.size() - i) / totalIndexSum;
            probabilities.push_back(probability);
        }

        // Randomly select one of the best scores with its respective row and column coordinates based on probabilities
        std::default_random_engine generator;
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
        index = distribution(generator);
        //emit logMessage("Metodo linear");
    } 
    else if (metodoEscolhaLocal == 2) {
        // Initialize gamma parameter
        double gamma = 1.0;

        // Calculate gravity weights
        std::vector<double> gravityWeights;
        double totalGravitySum = 0.0;

        for (const auto& scoreWithCoords : bestScoresWithCoords) {
            double score = std::get<0>(scoreWithCoords);
            double gravityWeight = 1.0 / std::pow(score, gamma); // Calculate weight using gravity function
            gravityWeights.push_back(gravityWeight);
            totalGravitySum += gravityWeight;
        }

        // Normalize the probabilities
        std::vector<double> probabilities;
        for (double weight : gravityWeights) {
            probabilities.push_back(weight / totalGravitySum);
        }

        // Randomly select one of the best scores with its respective row and column coordinates based on probabilities
        std::default_random_engine generator;
        std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());
        index = distribution(generator);
        //emit logMessage("Metodo Gravitacional");
    }
    else {
        index = 0;
    }
    // Get the selected best score and its coordinates
    auto selectedBestScoreWithCoords = bestScoresWithCoords[index];
    double selectedBestScore = std::get<0>(selectedBestScoreWithCoords);
    int selectedBestRow = std::get<1>(selectedBestScoreWithCoords);
    int selectedBestCol = std::get<2>(selectedBestScoreWithCoords);
    int selectedBestAccessPointID = std::get<3>(selectedBestScoreWithCoords);

    //emit logMessage("selectedBestAccessPointID = " + QString::number(selectedBestAccessPointID) + " selectedBestScore = " + QString::number(selectedBestScore) + " selectedBestRow = " + QString::number(selectedBestRow) + " selectedBestCol = " + QString::number(selectedBestCol));
    // Print information
    //emit logMessage("Agent " + QString::fromStdString(agent.name) + " at (" + QString::number(currentRow) + ", " + QString::number(currentCol) + ") moving to access point {" + QString::number(nodeGrid[selectedBestRow][selectedBestCol].id) + "} at (" + QString::number(selectedBestRow) + ", " + QString::number(selectedBestCol) + ") with score : " + QString::number(selectedBestScore));
    //std::cout << "Agent at (" << currentRow << ", " << currentCol << ") moving to access point at (" << bestrow << ", " << bestcol << ") with score: " << bestScore << std::endl;
    updateGridHeatPath(currentRow, currentCol, selectedBestRow, selectedBestCol, pathMap, gridHeatPath, gridHeatPathVector,alphaSteps,betaSteps);
    
    return std::make_tuple(selectedBestAccessPointID, selectedBestRow, selectedBestCol);
}

void ABM::moveAgents(std::vector<Agent>& agents, const std::vector<std::vector<Node>>& nodeGrid,const pathMap& pathMap, double alpha, double beta, double Maiorcaminho, double maxTrait, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, int alphaSteps, int betaSteps) {
    int row = 0;
    int col = 0;
    int nextAccessPointID = 0;
    for (auto& agent : agents) {
        // Calculate the next access point
        auto [nextAccessPointID, row, col] = calculateNextAccessPoint(agent, nodeGrid, pathMap, alpha, beta, Maiorcaminho, maxTrait, gridHeatPath, gridHeatPathVector, alphaSteps, betaSteps);

        // Move the agent to the new access point
        agent.currentAccessPointID = nextAccessPointID;
        agent.AccessPointIDValues.push_back(nextAccessPointID);
        agent.row = row;
        agent.col = col;
    }
}

void ABM::updateAgentTraits(std::vector<Agent>& agents, std::vector<std::vector<Node>>& nodeGrid, double pesoAgente, double pesoLugar, double minTrait, double maxTrait, std::vector<double>& GRIDentropyValues, std::vector<double>& AGENTSentropyValues) {
    int numerotraits = 0;
    std::unordered_map<int, std::pair<int, double>> accessPointStats;
    std::unordered_map<double, int> agentTraitFrequency; // For agent Shannon entropy calculation
    std::unordered_map<double, int> nodeTraitFrequency;  // For node Shannon entropy calculation
    int totalAgents = agents.size();
    int totalNodes = 0; // For Shannon entropy calculation

    // Initialize the unordered_map with agent counts and total traits at each access point
    for (const auto& agent : agents) {
        int accessPointID = agent.currentAccessPointID;
        accessPointStats[accessPointID].first += 1; // Increment count
        accessPointStats[accessPointID].second += agent.trait2; // Add trait to total
    }

    // Update traits based on neighbors
    for (auto& agent : agents) {
        Node& currentNode = nodeGrid[agent.row][agent.col];
        int accessPointID = agent.currentAccessPointID;

        // Get the total trait and count of neighbors at this access point
        auto& [neighborCount, totalTrait] = accessPointStats[accessPointID];

        if (neighborCount > 0) {
            double averageTrait = totalTrait / neighborCount;
            double placeTrait = currentNode.trait3;
            double nextAgentTrait = (agent.trait2 * (1.0 - pesoAgente)) + (placeTrait * pesoAgente);
            agent.trait2 = nextAgentTrait;
            agent.trait2Values.push_back(nextAgentTrait);
            numerotraits = agent.trait2Values.size();

            if (!currentNode.calculado) {
                currentNode.calculado = true;
                double nextPlaceTrait = (placeTrait * (1.0 - pesoLugar)) + (averageTrait * pesoLugar);
                currentNode.trait3 = nextPlaceTrait;
                currentNode.trait3Values.push_back(nextPlaceTrait);
            }
        }

        // Shannon entropy calculation for agents
        agentTraitFrequency[std::round(agent.trait2 * 10) / 10]++;
    }

    for (int i = 0; i < nodeGrid.size(); ++i) {
        for (int j = 0; j < nodeGrid[i].size(); ++j) {
            if (nodeGrid[i][j].accessPointID > 0) {
                if (nodeGrid[i][j].trait3Values.size() < numerotraits) {
                    // Repeat the last value of trait3Values until its size matches numerotraits
                    int lastValueIndex = nodeGrid[i][j].trait3Values.size() - 1;
                    double lastValue = nodeGrid[i][j].trait3Values[lastValueIndex];
                    while (nodeGrid[i][j].trait3Values.size() < numerotraits) {
                        nodeGrid[i][j].trait3Values.push_back(lastValue);
                    }
                }
                // Shannon entropy calculation for nodes
                nodeTraitFrequency[std::round(nodeGrid[i][j].trait3 * 10) / 10]++;
                totalNodes++;

                if (nodeGrid[i][j].calculado == true) {
                    nodeGrid[i][j].calculado = false;
                }
            }
        }
    }
    //emit agentesIniciados(agents);

    // Calculate Shannon entropy for agents
    double agentEntropy = 0.0;
    for (const auto& pair : agentTraitFrequency) {
        double probability = static_cast<double>(pair.second) / totalAgents;
        agentEntropy -= probability * std::log2(probability);
    }

    // Calculate Shannon entropy for nodes
    double nodeEntropy = 0.0;
    for (const auto& pair : nodeTraitFrequency) {
        double probability = static_cast<double>(pair.second) / totalNodes;
        nodeEntropy -= probability * std::log2(probability);
    }

    GRIDentropyValues.push_back(nodeEntropy);
    AGENTSentropyValues.push_back(agentEntropy);

    // Emit the entropy values or log them as needed
    //emit logMessage("Shannon Entropy of Agents: " + QString::number(agentEntropy));
    //emit logMessage("Shannon Entropy of Places: " + QString::number(nodeEntropy));

    //emit agentesIniciados(agents);
}




void ABM::savePaths(const pathMap& paths, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }

    int numEntries = static_cast<int>(paths.size());

    file.write(reinterpret_cast<const char*>(&numEntries), sizeof(int));

    for (const auto& entry : paths) {
        const auto& key = entry.first;
        file.write(reinterpret_cast<const char*>(&key), sizeof(std::tuple<int, int, int, int>));

        const PathInfo& pathInfo = entry.second;
        file.write(reinterpret_cast<const char*>(&pathInfo.cost), sizeof(double));

        int pathSize = static_cast<int>(pathInfo.path.size());
        file.write(reinterpret_cast<const char*>(&pathSize), sizeof(int));

        file.write(reinterpret_cast<const char*>(pathInfo.path.data()), pathSize * sizeof(std::pair<short, short>));

        file.write(reinterpret_cast<const char*>(&pathInfo.Length), sizeof(int));
        file.write(reinterpret_cast<const char*>(&pathInfo.totalVisitedNodes), sizeof(int));

        const std::pair<int, int>& source = pathInfo.source;
        const std::pair<int, int>& destination = pathInfo.destination;

        file.write(reinterpret_cast<const char*>(&source), sizeof(std::pair<int, int>));
        file.write(reinterpret_cast<const char*>(&destination), sizeof(std::pair<int, int>));

    }

    file.close();
}

pathMap ABM::loadPaths(const std::string& filename) {
    pathMap paths;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for reading." << std::endl;
        return paths;
    }

    int numEntries;
    file.read(reinterpret_cast<char*>(&numEntries), sizeof(int));
    std::cerr << "Numero de paths: " << numEntries << std::endl;

    for (int i = 0; i < numEntries; ++i) {
        std::tuple<int, int, int, int> key;
        file.read(reinterpret_cast<char*>(&key), sizeof(std::tuple<int, int, int, int>));

        PathInfo pathInfo;
        file.read(reinterpret_cast<char*>(&pathInfo.cost), sizeof(double));

        int pathSize;
        file.read(reinterpret_cast<char*>(&pathSize), sizeof(int));

        pathInfo.path.resize(pathSize);
        file.read(reinterpret_cast<char*>(pathInfo.path.data()), pathSize * sizeof(std::pair<short, short>));

        file.read(reinterpret_cast<char*>(&pathInfo.Length), sizeof(int));
        file.read(reinterpret_cast<char*>(&pathInfo.totalVisitedNodes), sizeof(int));
        //std::cout << "Path: ";
        //for (const auto& point : pathInfo.path) {
        //    std::cout << "(" << point.first << ", " << point.second << ") ";
        //}
        //std::cout << std::endl;

        std::pair<int, int> source, destination;
        file.read(reinterpret_cast<char*>(&source), sizeof(std::pair<int, int>));
        file.read(reinterpret_cast<char*>(&destination), sizeof(std::pair<int, int>));

        pathInfo.source = source;
        pathInfo.destination = destination;

        paths[key] = pathInfo;
    }

    file.close();

    return paths;
}

void ABM::addNeighborsToQueueAStar(
    const std::vector<std::vector<Node>>& nodeGrid,
    std::priority_queue<PathNode, std::vector<PathNode>, ComparePathNode>& pq,
    std::map<std::pair<int, int>, PathNode>& nodesMap,
    std::vector<bool>& visited,
    int row, int col,
    double costSoFar,
    int parentRow, int parentCol,
    int destRow, int destCol) {

    int numRows = static_cast<int>(nodeGrid.size());
    int numCols = numRows > 0 ? static_cast<int>(nodeGrid[0].size()) : 0;

    std::vector<std::pair<int, int>> directions = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}, // Cardinal directions
        {1, 1}, {-1, -1}, {1, -1}, {-1, 1} // Diagonal directions
    };

    for (const auto& dir : directions) {
        int newRow = row + dir.first;
        int newCol = col + dir.second;

        // Check if the new position is within bounds
        if (newRow >= 0 && newRow < numRows && newCol >= 0 && newCol < numCols &&
            !visited[newRow * numCols + newCol] && nodeGrid[newRow][newCol].trait3 != -1 &&
            (nodeGrid[newRow][newCol].trait3 == 0 || nodeGrid[newRow][newCol].accessPointID > 0 )) {

            double stepCost = (dir.first == 0 || dir.second == 0) ? 1.0 : 1.414; // Adjust cost for diagonal movements
            double newCost = costSoFar + stepCost;
            double heuristic = std::max(std::abs(newRow - destRow), std::abs(newCol - destCol)); // Using Chebyshev distance

            PathNode newNode(newRow, newCol, newCost, heuristic, row, col);

            if (!nodesMap.count({ newRow, newCol }) || newCost < nodesMap[{newRow, newCol}].costSoFar) {
                pq.push(newNode);
                nodesMap[{newRow, newCol}] = newNode;
            }
        }
    }
}

void ABM::calculateShortestPathsAStar(
    const std::vector<std::vector<Node>>& nodeGrid,
    int startRow, int startCol,
    int destRow, int destCol,
    pathMap& pathMap) {

    std::priority_queue<PathNode, std::vector<PathNode>, ComparePathNode> pq;
    std::map<std::pair<int, int>, PathNode> nodesMap;
    std::vector<bool> visited(nodeGrid.size() * nodeGrid[0].size(), false);

    PathNode startNode(startRow, startCol, 0, std::max(std::abs(startRow - destRow), std::abs(startCol - destCol)), -1, -1);
    pq.push(startNode);
    nodesMap[{startRow, startCol}] = startNode;
    int timeLimitMilliseconds = 60000;

    using Clock = std::chrono::steady_clock;
    auto startTime = Clock::now(); // Capture the start time

    while (!pq.empty()) {
        auto currentTime = Clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
        if (elapsedTime > timeLimitMilliseconds) {
            emit logMessage("Path from (" + QString::number(startRow) + " , " + QString::number(startCol) + ") to ( " + QString::number(destRow) + " , " + QString::number(destCol) + ") could not be found. Time limit reached.");
            pathMap[{startRow, startCol, destRow, destCol}].Length = std::numeric_limits<int>::max();
            return; // Or handle the time limit exceedance as appropriate for your application
        }
        PathNode current = pq.top();
        pq.pop();

        int currentIndex = current.row * nodeGrid[0].size() + current.col;
        if (visited[currentIndex]) continue;
        visited[currentIndex] = true;

        if (current.row == destRow && current.col == destCol) {
            // Path reconstruction
            std::vector<std::pair<short, short>> path;
            PathNode node = current;
            while (node.parentRow != -1 && node.parentCol != -1) {
                path.emplace_back(node.row, node.col);
                node = nodesMap[{node.parentRow, node.parentCol}];
            }
            path.emplace_back(startRow, startCol); // Adding the start position
            std::reverse(path.begin(), path.end());

            // Store the path
            pathMap[{startRow, startCol, destRow, destCol}].path = path;
            break;
        }

        // Add neighboring nodes to the queue and map
        addNeighborsToQueueAStar(nodeGrid, pq, nodesMap, visited, current.row, current.col, current.costSoFar, current.row, current.col, destRow, destCol);
    }
    //generatePathImage(nodeGrid, pathMap[{startRow, startCol, destRow, destCol}].path, "(" + std::to_string(startRow) + " , " + std::to_string(startCol) + " , " + std::to_string(destRow) + " , " + std::to_string(destCol) + ")");
}



void ABM::generatePathImage(const std::vector<std::vector<Node>>& nodeGrid, const std::vector<std::pair<int, int>>& path, std::string title) {
    // Get the size of the node grid
    int numRows = static_cast<int>(nodeGrid.size());
    int numCols = numRows > 0 ? static_cast<int>(nodeGrid[0].size()) : 0;

    // Create a blank image with white pixels
    cv::Mat image(numRows, numCols, CV_8UC3, cv::Scalar(255, 255, 255));

    // Set the color of pixels corresponding to path nodes to red
    for (const auto& node : path) {
        int row = node.first;
        int col = node.second;
        image.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255); // BGR color format: Red
    }

    // Specify the directory where you want to save the image
    std::string directoryName = "path_images"; // Example directory name
    std::filesystem::path dir(directoryName);
    std::filesystem::path filePath = dir / (title + ".png");

    // Check if the directory exists; if not, create it
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir); // Create the directory
    }

    // Save the image to the specified path
    cv::imwrite(filePath.string(), image);
}



std::vector<std::pair<int, int>> ABM::reconstructShortestPath(pathMap& pathMap, int startRow, int startCol, int destRow, int destCol) {
    std::vector<std::pair<int, int>> shortestPath;

    // Start from the destination node
    int currentRow = destRow;
    int currentCol = destCol;

    // Retrieve the corresponding pathMap entry for the destination
    const PathInfo& pathInfo = pathMap.at({ startRow, startCol, destRow, destCol });

    // Get the path vector containing visited nodes
    const std::vector<std::pair<short, short>>& path = pathInfo.path;

    // Traverse the path in reverse order to reconstruct the shortest path
    for (auto it = path.rbegin(); it != path.rend(); ++it) {
        // Add the current node to the shortest path
        shortestPath.push_back(*it);

        // Move to the previous node
        currentRow = it->first;
        currentCol = it->second;
    }

    // Add the starting node to complete the shortest path
    shortestPath.push_back({ startRow, startCol });

    return shortestPath;
}


// Modify the calculateShortestPaths function to determine source and destination
std::vector<std::vector<PathInfo>> ABM::calculateShortestPaths(const std::vector<std::vector<Node>>& nodeGrid, pathMap& mainPathMap) {
    int numRows = static_cast<int>(nodeGrid.size());
    int numCols = numRows > 0 ? static_cast<int>(nodeGrid[0].size()) : 0;
    std::vector<std::vector<PathInfo>> shortestPaths(numRows, std::vector<PathInfo>(numCols));

    // Determine the number of available cores
    //int numCores = QThread::idealThreadCount()-4;
    int numCores = threadsLocal;
    emit logMessage("Numero de Threads utilizados : " + QString::number(numCores * 2));

    // Calculate the number of rows each thread should process
    int rowsPerThread = numRows / numCores;
    int extraRows = numRows % numCores;  // Extra rows that need to be distributed

    QVector<QFuture<pathMap>> futures;
    // Set to store visited pairs of source and destination nodes
    std::set<std::tuple<int, int, int, int>> visitedPairs;

    // In your calculateShortestPaths function, before launching threads
    QAtomicInt completedTasks(0);
    QAtomicInt completedTasks2(0);
    int completed2 = 0;
    //int totalTasks = numRows * numCols; // Adjust based on your actual total tasks
    timer.start();
    float progress = 0;
    int startRow = 0;
    for (int i = 0; i < numCores; ++i) {
        // Determine the exact number of rows for this thread to process
        int endRow = startRow + rowsPerThread + (i < extraRows ? 1 : 0);
        int rowsToProcessHere = rowsPerThread + (i < extraRows ? 1 : 0);

        // Launch a thread to process a subset of rows
        auto future = QtConcurrent::run ([this, &nodeGrid, &completedTasks,&visitedPairs, &completedTasks2, progress, completed2, rowsToProcessHere, numRows, startRow, endRow, numCols]() -> pathMap {
            pathMap localPathMap;
            for (int row = startRow; row < endRow; ++row) {
                for (int col = 0; col < numCols; ++col) {
                    if (nodeGrid[row][col].accessPointID > 0) {
                        for (int destRow = 0; destRow < numRows; ++destRow) {
                            for (int destCol = 0; destCol < numCols; ++destCol) {
                                if (!(nodeGrid[destRow][destCol].accessPointID > 0) || (row == destRow && col == destCol)) continue;
                                // Check if the reverse pair has already been visited
                                if (visitedPairs.count(std::make_tuple(destRow, destCol, row, col)) > 0) {
                                    continue;  // Skip if already visited
                                }
                                this->calculateShortestPathsAStar(nodeGrid, row, col, destRow, destCol, localPathMap);
                                int completed2 = completedTasks2.fetchAndAddRelaxed(1) + 1;
                                float progress = static_cast<float>(completed2) / totalTasks;
                                //float progress = static_cast<float>(completed) / totalTasks;
                                // Mark the pair as visited
                                visitedPairs.insert(std::make_tuple(row, col, destRow, destCol));
                                emit progressUpdated2(progress, QString::number(completed2) + " de " + QString::number(totalTasks));
                                
                            }
                            
                        }
                    }
                    
                    
                }
                //completedTasks.fetchAndAddRelaxed(rowsToProcessHere * numCols); // Assuming each row has numCols tasks
            }
            return localPathMap;
        });

        futures.push_back(future);

        startRow = endRow;  // Update startRow for the next batch
    }

    // Wait for all threads to complete and merge the results
    for (auto& future : futures) {
        future.waitForFinished(); // Ensure the task is finished
        pathMap result = future.result();  // Get the result of the future

        for (const auto& entry : result) {
            // Here we simply overwrite any existing entry in mainPathMap with the new one
            // You might want to handle conflicts differently depending on your requirements
            mainPathMap[entry.first] = entry.second;
        }
    }
    //timer.stop(); // Stop the timer after all tasks are completed
    emit progressUpdated(1.0f); // Ensure 100% progress is reported at the end

    // After all threads complete, you may want to convert the mainPathMap to the shortestPaths format as required.

    return shortestPaths;
}

int ABM::calculateUniquePairs(int n) {
    // Calculate the total number of cells in the grid
    int totalCells = static_cast<int>(n) * n;

    // Calculate the number of unique pairs, excluding reverse pairs and self-pairs
    // This uses the simplified combination formula for k = 2: C(n, 2) = n * (n - 1) / 2
    int uniquePairs = n * (n - 1) / 2;

    return uniquePairs;
}

void ABM::saveNodeGrid(const std::vector<std::vector<Node>>& nodeGrid, const std::string& filename, int numRows, int numCols) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for writing." << std::endl;
        return;
    }

    // Save numRows and numCols
    file.write(reinterpret_cast<const char*>(&numRows), sizeof(int));
    file.write(reinterpret_cast<const char*>(&numCols), sizeof(int));

    // Save Node objects
    for (const auto& row : nodeGrid) {
        for (const auto& node : row) {
            // Save individual members of Node
            file.write(reinterpret_cast<const char*>(&node.id), sizeof(int));
            file.write(reinterpret_cast<const char*>(&node.trait3), sizeof(double));
            // Save trait3Values
            int trait3ValuesSize = static_cast<int>(node.trait3Values.size());
            file.write(reinterpret_cast<const char*>(&trait3ValuesSize), sizeof(int));
            file.write(reinterpret_cast<const char*>(node.trait3Values.data()), trait3ValuesSize * sizeof(double));
            // Save other members of Node
            file.write(reinterpret_cast<const char*>(&node.row), sizeof(int));
            file.write(reinterpret_cast<const char*>(&node.col), sizeof(int));
            file.write(reinterpret_cast<const char*>(&node.analysed), sizeof(bool));
            file.write(reinterpret_cast<const char*>(&node.isPath), sizeof(bool));
            file.write(reinterpret_cast<const char*>(&node.accessPointID), sizeof(int));
        }
    }

    file.close();
}

std::vector<std::vector<Node>> ABM::loadNodeGrid(const std::string& filename) {
    std::vector<std::vector<Node>> nodeGrid;

    int numRows = 0;
    int numCols = 0;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file for reading." << std::endl;
        return nodeGrid;
    }

    // Read numRows and numCols
    file.read(reinterpret_cast<char*>(&numRows), sizeof(int));
    file.read(reinterpret_cast<char*>(&numCols), sizeof(int));

    nodeGrid.resize(numRows, std::vector<Node>(numCols));

    // Read Node objects
    for (auto& row : nodeGrid) {
        for (auto& node : row) {
            // Read individual members of Node
            file.read(reinterpret_cast<char*>(&node.id), sizeof(int));
            file.read(reinterpret_cast<char*>(&node.trait3), sizeof(double));
            // Read trait3Values
            int trait3ValuesSize;
            file.read(reinterpret_cast<char*>(&trait3ValuesSize), sizeof(int));
            node.trait3Values.resize(trait3ValuesSize);
            file.read(reinterpret_cast<char*>(node.trait3Values.data()), trait3ValuesSize * sizeof(double));
            // Read other members of Node
            file.read(reinterpret_cast<char*>(&node.row), sizeof(int));
            file.read(reinterpret_cast<char*>(&node.col), sizeof(int));
            file.read(reinterpret_cast<char*>(&node.analysed), sizeof(bool));
            file.read(reinterpret_cast<char*>(&node.isPath), sizeof(bool));
            file.read(reinterpret_cast<char*>(&node.accessPointID), sizeof(int));
        }
    }

    file.close();

    return nodeGrid;
}

void ABM::processMap(std::vector<std::vector<Node>>& nodeGrid, int numRows, int numCols, pathMap& pathMap, std::string filenameToSave, int option, int numAgents,int interactions,double alpha,double beta,double pesoAgente,double pesoLugar, int steps, double stepSize, std::vector<PolygonData> poligonos) {
    // Set the width of the loading bar
    const int loadingBarWidth = 50;

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            // Identify paths (nodes with trait = 0)
            nodeGrid[i][j].isPath = (nodeGrid[i][j].trait3 == 0);

            // Update the loading bar
            float progress = static_cast<float>(i * numCols + j + 1) / (numRows * numCols);
            int barWidth = static_cast<int>(loadingBarWidth * progress);

            emit progressUpdated(progress);
            // Print the loading bar
            //std::cout << "\rProgress: [" << std::string(barWidth, '#') << std::string(loadingBarWidth - barWidth, ' ') << "] "
            //    << std::fixed << std::setprecision(2) << progress * 100.0 << "%";
            //std::cout.flush();
        }
    }

    // Print a new line after completing the loading bar
    //std::cout << std::endl;

    // Identify places and determine access points
    std::unordered_map<int, std::pair<int, int>> placeAccessPoints;  // ID -> (row, col) of access point
    std::unordered_set<int> uniqueIDs;

    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            uniqueIDs.insert(nodeGrid[i][j].id);

            if (!nodeGrid[i][j].isPath) {
                // Check if we already found an access point for this place/ID
                if (placeAccessPoints.find(nodeGrid[i][j].id) == placeAccessPoints.end()) {
                    bool hasPathNeighbor = false;

                    // Iterate through neighbors to find the access point for the place
                    for (int ni = -1; ni <= 1; ++ni) {
                        for (int nj = -1; nj <= 1; ++nj) {
                            int neighborRow = i + ni;
                            int neighborCol = j + nj;

                            if (neighborRow >= 0 && neighborRow < numRows &&
                                neighborCol >= 0 && neighborCol < numCols &&
                                nodeGrid[neighborRow][neighborCol].isPath) {
                                // Found a neighbor that is a path
                                hasPathNeighbor = true;
                                break;
                            }
                        }
                        if (hasPathNeighbor) {
                            break;  // No need to check other neighbors
                        }
                    }

                    if (hasPathNeighbor) {
                        // Set the non-path node as the access point
                        placeAccessPoints[nodeGrid[i][j].id] = { i, j };
                        //std::cout << "Access point found for ID " << nodeGrid[i][j].id << " at (" << i << ", " << j << ")" << std::endl;
                    }
                }
            }
        }
    }


    int numUniqueIDs = static_cast<int>(uniqueIDs.size());
    //std::cout << "Number of unique IDs in the nodeGrid: " << numUniqueIDs << std::endl;
    emit logMessage("Number of unique IDs in the nodeGrid: " + QString::number(numUniqueIDs));

    // Print the number of access points and their locations
    //std::cout << "Number of Access Points: " << placeAccessPoints.size() << std::endl;
    totalTasks = calculateUniquePairs(placeAccessPoints.size());
    emit logMessage("Number of Access Points: " + QString::number(placeAccessPoints.size()));
    //std::cout << "Access Point Locations:" << std::endl;
    //for (const auto& pair : placeAccessPoints) {
    //    int id = pair.first;
    //    int accessPointRow = pair.second.first;
    //    int accessPointCol = pair.second.second;
    //    std::cout << "Access Point ID " << id << ": (" << accessPointRow << ", " << accessPointCol << ")" << std::endl;
    //}

    // Store access point information in nodes
    int IDs = 0;
    for (auto& pair : placeAccessPoints) {
        ++IDs;
        int id = pair.first;
        int accessPointRow = pair.second.first;
        int accessPointCol = pair.second.second;

        nodeGrid[accessPointRow][accessPointCol].accessPointID = IDs;
        //std::cout << "E nodeGrid ( " << accessPointRow << ", " << accessPointCol << ")." << std::endl;
    }

    // Prompt the user for the filename to save the updated nodeGrid
    //std::string filenameToSave;
    //std::cout << "Enter the filename to save the nodeGrid: ";
    //std::cin >> filenameToSave;

    // Save the updated nodeGrid to the binary file
    saveNodeGrid(nodeGrid, filenameToSave + ".nodes", numRows, numCols);
    createImageFromNodeGrid(nodeGrid, filenameToSave);
    //exportImageFromNodeGrid(nodeGrid);
    // Calculate and save A* shortest paths
    std::vector<std::vector<PathInfo>> shortestPaths = calculateShortestPaths(nodeGrid, pathMap);
    savePaths(pathMap, filenameToSave + ".paths");
    savePolygonsToFile(poligonos, filenameToSave + ".polygons");
    // Call the function to save paths to a CSV file
    //savePathsToCSV(filenameToSave + "_Paths.csv", pathMap);
    // Run the simulation
    chooseRunMethod(nodeGrid, pathMap, filenameToSave, option, numAgents, interactions, alpha, beta, pesoAgente, pesoLugar, steps, stepSize, poligonos);


    //std::cout << "Processing completed successfully." << std::endl;
}

void ABM::printNodeGrid(const std::vector<std::vector<Node>>& nodeGrid, int numRows, int numCols) {
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << "Node[" << i << "][" << j << "]: "
                << "ID=" << nodeGrid[i][j].id << ", "
                << "Analysed=" << (nodeGrid[i][j].analysed ? "true" : "false")
                << std::endl;
        }
    }
}

std::vector<std::string> ABM::pegarFields(const char* shapefilePath) {
    GDALAllRegister();
    pathMap pathMap;
    GDALDataset* poDS = (GDALDataset*)GDALOpenEx(shapefilePath, GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        std::cerr << "Error: Unable to open shapefile." << std::endl;
    }
    // Access the first layer in the shapefile
    OGRLayer* poLayer = poDS->GetLayer(0);
    std::vector<std::string> fieldNames;
    // Iterate over fields and retrieve their names
    OGRFeatureDefn* poFDefn = poLayer->GetLayerDefn();
    int numFields = poFDefn->GetFieldCount();
    for (int i = 0; i < numFields; ++i) {
        OGRFieldDefn* poFieldDefn = poFDefn->GetFieldDefn(i);
        if (poFieldDefn != NULL) {
            const char* fieldName = poFieldDefn->GetNameRef();
            fieldNames.push_back(fieldName);
        }
    }
    return fieldNames;
}

void ABM::processShapefile(const char* shapefilePath, const char* shapefilePath2, const std::string filenameToSave, std::string CAMINHODAPASTA, double nodeSize, std::string NOMEDOUSO, int option, int numAgents, int interactions, double alpha, double beta, double pesoAgente, double pesoLugar, int steps, double stepSize, bool node, int VoltarPraCasa, int DistribuicaoUsos, int Maxtraits, int MaxtraitsAgents, int DistribuicaoAgentes, int threads, int metodoEscolha, std::unordered_map<int, int> idTraitMap, std::vector<std::pair<int, int>> agentTraits) {
    GDALAllRegister();
    pathMap pathMap;
    VoltarPraCasaLocal = VoltarPraCasa;
    DistribuicaoUsosLocal = DistribuicaoUsos;
    MaxtraitsLocal = Maxtraits;
    MaxtraitsAgentsLocal = MaxtraitsAgents;
    DistribuicaoAgentesLocal = DistribuicaoAgentes;
    threadsLocal = threads;
    metodoEscolhaLocal = metodoEscolha;
    idTraitMapLocal = idTraitMap;
    agentTraitsLocal = agentTraits;

    GDALDataset* poDS = (GDALDataset*)GDALOpenEx(shapefilePath, GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        std::cerr << "Error: Unable to open shapefile." << std::endl;
        return;
    }

    // Access the first layer in the shapefile
    OGRLayer* poLayer = poDS->GetLayer(0);

    // Get the envelope (bounding box) of the shapefile
    OGREnvelope envelope;
    poLayer->GetExtent(&envelope);

    // Find the index of the field
    OGRFeatureDefn* poFDefn = poLayer->GetLayerDefn();
    int fieldIndex = poFDefn->GetFieldIndex(NOMEDOUSO.c_str());


    if (fieldIndex == -1) {
        std::cerr << "Error: Field not found in the shapefile." << std::endl;
        GDALClose(poDS);
    }

    
    // Calculate the size of the envelope
    double envelopeWidth = envelope.MaxX - envelope.MinX;
    double envelopeHeight = envelope.MaxY - envelope.MinY;

    // Get the spatial reference system (SRS) of the layer
    OGRSpatialReference* spatialRef = poLayer->GetSpatialRef();
    const char* unitsName = spatialRef->GetAttrValue("UNIT");
    double unitsConversionFactor = spatialRef->GetLinearUnits(); // Factor to convert to meters


    // Convert to meters if the units are not already in meters
    if (std::string(unitsName) != "metre" && unitsConversionFactor != 1.0) {
        envelopeWidth *= unitsConversionFactor;
        envelopeHeight *= unitsConversionFactor;
    }

    emit logMessage("Largura: " + QString::number(envelopeWidth));
    emit logMessage("Altura: " + QString::number(envelopeHeight));


    // Calculate the number of rows and columns based on the node size
    int numRows = static_cast<int>(envelopeHeight / nodeSize);
    int numCols = static_cast<int>(envelopeWidth / nodeSize);

    // Create the nodeGrid based on the calculated rows and columns
    std::vector<std::vector<Node>> nodeGrid(numRows, std::vector<Node>(numCols));

    if (node) {
        nodeGrid = loadNodeGrid(shapefilePath);
    }
    else {
        double stepX = (envelope.MaxX - envelope.MinX) / numCols;
        double stepY = (envelope.MaxY - envelope.MinY) / numRows;

        for (int i = 0; i < numRows; ++i) {
            double nodeY = envelope.MinY + i * stepY;

            for (int j = 0; j < numCols; ++j) {
                double nodeX = envelope.MinX + j * stepX;

                // Use SetSpatialFilterRect to filter features within a bounding box
                poLayer->SetSpatialFilterRect(nodeX, nodeY, nodeX + stepX, nodeY + stepY);

                // Iterate through the features that intersect with the bounding box
                poLayer->ResetReading();
                OGRFeature* poFeature = poLayer->GetNextFeature();

                if (poFeature != nullptr) {
                    // Assign properties to the node
                    int commonID = static_cast<int>(poFeature->GetFID());
                    nodeGrid[i][j].id = commonID;
                    nodeGrid[i][j].trait3 = poFeature->GetFieldAsDouble(fieldIndex);
                    nodeGrid[i][j].trait3Values.push_back(poFeature->GetFieldAsDouble(fieldIndex));
                    nodeGrid[i][j].analysed = false;
                    nodeGrid[i][j].row = i;
                    nodeGrid[i][j].col = j;
                    OGRFeature::DestroyFeature(poFeature);
                }

                // Update the loading bar
                float progress = static_cast<float>(i * numCols + j + 1) / (numRows * numCols);
                emit progressUpdated(progress);
            }
        }

        GDALClose(poDS);
    }

    // Print a new line after completing the loading bar
    //std::cout << std::endl;

    // Prompt the user for the filename to save the nodeGrid
    //std::string filenameToSave;
    //std::cout << "Enter the filename to save the nodeGrid: ";
    //std::cin >> filenameToSave;

    // Save the nodeGrid to the binary file
    //saveNodeGrid(nodeGrid, filenameToSave, numRows, numCols);

    // Print the loaded nodeGrid
    //printNodeGrid(nodeGrid, numRows, numCols);
    //std::string filenameToSave = extractFilename(std::string (shapefilePath));

    //std::cout << filenameToSave << std::endl;

    //std::string filenameToSave2 = extractFilename(filenameToSave);

    // Convert the const char* to std::string
    std::string fullPathStr = filenameToSave;

    std::string filenameToSave2; 

    // Find the position of the last '\' character
    size_t lastSlashPos = fullPathStr.find_last_of("/");
    if (lastSlashPos == std::string::npos) {
        // No '\' character found, return empty string or fullPath as per requirement
        std::cerr << "No '/' character found." << std::endl;
        filenameToSave2 = "";
    }

    // Find the position of the last '.' character
    size_t lastDotPos = fullPathStr.find_last_of(".");
    if (lastDotPos == std::string::npos || lastDotPos < lastSlashPos) {
        // No '.' character found after the last '\', or '.' character is before '\', return empty string
        std::cerr << "No '.' character found after the last '/', or '.' character is before '/'." << std::endl;
        filenameToSave2 = "";
    }

    // Extract the substring between the last '\' and the last '.'
    filenameToSave2 = fullPathStr.substr(lastSlashPos + 1, lastDotPos - lastSlashPos - 1);

    std::vector<PolygonData> poligonos = criarPoligonos(shapefilePath2, NOMEDOUSO);

    emit polygonsReady(poligonos);

    processMap(nodeGrid, numRows, numCols, pathMap, CAMINHODAPASTA + "/" + filenameToSave2, option, numAgents, interactions, alpha, beta, pesoAgente, pesoLugar, steps, stepSize, poligonos);

    //std::cout << "Processing completed successfully." << std::endl;
    emit logMessage("Processing completed successfully." );
    //emit ImageProduced(QString::fromStdString(CAMINHODAPASTA + "/" + filenameToSave2));
     
    //return nodeGrid;
}


QColor ABM::getColorForValue(double value, double highestLabelValue) {
    // Calculate step size between gradient stops
    qreal step = 1.0 / highestLabelValue;

    // Generate gradient stops dynamically based on highest label value
    QGradientStops gradientStops;
    for (int i = 0; i <= highestLabelValue; ++i) {
        gradientStops.push_back({ static_cast<qreal>(i) * step, QColor::fromHsvF(static_cast<qreal>(i) * step, 1.0, 1.0) });
    }

    // Find the closest gradient stop for the given value
    qreal t = static_cast<qreal>(value) / highestLabelValue;
    auto lowerStop = gradientStops.begin();
    for (auto it = gradientStops.begin(); it != gradientStops.end(); ++it) {
        if (it->first >= t) {
            break;
        }
        lowerStop = it;
    }

    // Interpolate colors between the two adjacent stops
    auto upperStop = lowerStop + 1;
    if (upperStop == gradientStops.end()) {
        return lowerStop->second;
    }
    qreal lowerT = lowerStop->first;
    qreal upperT = upperStop->first;
    QColor lowerColor = lowerStop->second;
    QColor upperColor = upperStop->second;
    qreal interpolationFactor = (t - lowerT) / (upperT - lowerT);
    return lowerColor.toHsv().toRgb().toRgb().lighter(100 * interpolationFactor);
}


std::vector<PolygonData> ABM::criarPoligonos(const char* shapefilePath, std::string NOMEDOUSO) {
    GDALAllRegister();

    // Open the shapefile
    GDALDataset* poDS = (GDALDataset*)GDALOpenEx(shapefilePath, GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) {
        std::cerr << "Error: Unable to open shapefile." << std::endl;
    }

    // Access the first layer in the shapefile
    OGRLayer* poLayer = poDS->GetLayer(0);
    OGRFeatureDefn* poFDefn = poLayer->GetLayerDefn();

    std::vector<PolygonData> polygons;

    int fieldIndex = poFDefn->GetFieldIndex(NOMEDOUSO.c_str());

    //int highestLabelValue = 11; // Variable to store the highest label value

    // Loop through features in the layer
    OGRFeature* poFeature;
    //int id = 0;
    while ((poFeature = poLayer->GetNextFeature()) != NULL) {
        // Extract geometry from the feature

        int labelValue = poFeature->GetFieldAsInteger(fieldIndex);

        OGRGeometry* poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPolygon) {
            PolygonData polygonData;

            int commonID = poFeature->GetFID(); // Extract common ID from shapefile
            // Get unique identifier for the polygon
            polygonData.id = commonID;

            // Extract points of the polygon
            OGRPolygon* poPolygon = (OGRPolygon*)poGeometry;
            OGRLinearRing* poRing = poPolygon->getExteriorRing();
            for (int i = 0; i < poRing->getNumPoints(); ++i) {
                double x = poRing->getX(i);
                double y = poRing->getY(i);
                polygonData.points.emplace_back(x, y);
            }

            
            //polygonData.color = getColorForValue(labelValue, highestLabelValue);
            polygonData.color = QColor("white");

            // Add polygon data to the vector
            polygons.push_back(polygonData);
        }

        // Destroy the feature
        OGRFeature::DestroyFeature(poFeature);
    }

    // Close the shapefile
    GDALClose(poDS);

    //polygonsLocalCore = polygons;
    //logMessage(QString::number(polygonsLocalCore.size()));
    // Emit signal with polygon data
    return polygons;
}

void ABM::chooseRunMethod(std::vector<std::vector<Node>>& nodeGrid, pathMap& pathMap, std::string filenameToSave, int option, int numAgents,int interactions,double alpha,double beta,double pesoAgente,double pesoLugar, int steps, double stepSize, std::vector<PolygonData> poligonos) {
    //std::cout << "Choose an option:" << std::endl;
    //std::cout << "1. One Simulation" << std::endl;
    //std::cout << "2. multiple simulations" << std::endl;
    

    //int option;
    //std::cin >> option;
    std::vector<Agent> agents;

    GridHeatPath gridHeatPath = convertPolygonsToGrid(poligonos, nodeGrid[0].size(), nodeGrid[1].size());
    std::vector<std::vector<GridHeatPath>> gridHeatPathVector(steps, std::vector<GridHeatPath>(steps));
    std::vector<std::vector<std::vector<Agent>>> AgentsVector(steps, std::vector <std::vector<Agent>>(steps));

    auto [minTrait, maxTrait] = findTraitBounds(nodeGrid);
    std::vector<QColor> colorVector;

    accessPointMap.clear();

    // Loop through the nodeGrid using its sizes
    for (size_t i = 0; i < nodeGrid.size(); ++i) {
        for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
            int accessPointID = nodeGrid[i][j].accessPointID;
            int row = nodeGrid[i][j].row;
            int col = nodeGrid[i][j].col;

            // Store the (row, col) pair in the map with accessPointID as the key
            accessPointMap[accessPointID] = std::make_pair(row, col);
        }
    }

    if (DistribuicaoUsosLocal == 1) {
        std::unordered_map<int, int> idToRandomValueMap;
        std::unordered_map<int, std::unordered_set<int>> traitIdSetMap;  // Map to track unique ids for each trait

        for (size_t i = 0; i < nodeGrid.size(); ++i) {
            for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
                int currentId = nodeGrid[i][j].id;

                if (nodeGrid[i][j].trait3 != -1) {
                    // Check if the current id already has a random value assigned
                    if (idToRandomValueMap.find(currentId) == idToRandomValueMap.end()) {
                        // Generate a new unique random value for this id
                        int randomValue = static_cast<int>(std::round(generateRandomValue(minTrait, MaxtraitsLocal)));
                        idToRandomValueMap[currentId] = randomValue;
                    }

                    // Assign the same random value to all nodes with the same id
                    int assignedTrait = idToRandomValueMap[currentId];
                    nodeGrid[i][j].trait3 = assignedTrait;
                    nodeGrid[i][j].trait3Values.clear(); // Clear existing values (if necessary)
                    nodeGrid[i][j].trait3Values.push_back(assignedTrait); // Add the random value

                    // Track unique ids for each trait
                    traitIdSetMap[assignedTrait].insert(currentId);
                }
            }
        }

        // Log the count of each trait value, excluding 0 and -1
        for (const auto& [trait, idSet] : traitIdSetMap) {
            if (trait != 0 && trait != -1) {
                emit logMessage(QString("Trait %1: %2 ids").arg(trait).arg(idSet.size()));
            }
        }

        // Update the values of minTrait and maxTrait after modifying nodeGrid
        std::tie(minTrait, maxTrait) = findTraitBounds(nodeGrid);
    }
    else if (DistribuicaoUsosLocal == 2) {
        // Group nodes by their id, excluding ids 0 and -1
        std::unordered_map<int, std::vector<std::pair<int, int>>> groupedNodes;

        for (size_t i = 0; i < nodeGrid.size(); ++i) {
            for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
                int currentId = nodeGrid[i][j].id;
                if (currentId != 0 && currentId != -1 && nodeGrid[i][j].trait3 != -1) {
                    groupedNodes[currentId].emplace_back(i, j);
                }
            }
        }

        // Create a vector of keys (group IDs) for shuffling
        std::vector<int> groupKeys;
        for (const auto& group : groupedNodes) {
            groupKeys.push_back(group.first);
        }

        // Shuffle the group keys
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(groupKeys.begin(), groupKeys.end(), g);

        // Determine the number of unique ids and the traits to distribute
        int numTraits = static_cast<int>(MaxtraitsLocal - minTrait + 1);
        std::vector<int> traitValues;
        for (int i = minTrait; i <= MaxtraitsLocal; ++i) {
            if (i != 0 && i != -1) {
                traitValues.push_back(i);
            }
        }

        // Map to track the count of each trait value assigned
        std::unordered_map<int, int> traitValueCounts;

        // Distribute trait values uniformly across the shuffled group keys
        int traitIndex = 0;
        for (int groupKey : groupKeys) {
            auto& positions = groupedNodes[groupKey];

            // Assign a trait value to all nodes with the same id (shuffled order)
            int assignedTrait = traitValues[traitIndex];
            for (auto& [i, j] : positions) {
                nodeGrid[i][j].trait3 = assignedTrait;
                nodeGrid[i][j].trait3Values.clear();
                nodeGrid[i][j].trait3Values.push_back(assignedTrait);
            }

            // Increment the count for this trait value
            traitValueCounts[assignedTrait]++;

            // Move to the next trait value
            traitIndex = (traitIndex + 1) % traitValues.size();
        }

        // Log the count of each trait value, excluding 0 and -1
        for (const auto& [trait, count] : traitValueCounts) {
            emit logMessage(QString("Trait %1: %2 ids").arg(trait).arg(count));
        }

        // Update the values of minTrait and maxTrait after modifying nodeGrid
        std::tie(minTrait, maxTrait) = findTraitBounds(nodeGrid);
    }
    else if (DistribuicaoUsosLocal == 3) {
        emit logMessage(QString("chegou aqui"));
        for (size_t i = 0; i < nodeGrid.size(); ++i) {
            for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
                int nodeId = nodeGrid[i][j].id;

                // Check if the idTraitMapLocal has an entry for this node ID
                if (idTraitMapLocal.find(nodeId) != idTraitMapLocal.end()) {
                    // Set the trait3 value with the corresponding value from idTraitMapLocal
                    nodeGrid[i][j].trait3 = idTraitMapLocal[nodeId];
                    nodeGrid[i][j].trait3Values.clear();
                    nodeGrid[i][j].trait3Values.push_back(nodeGrid[i][j].trait3);
                }
                else {
                    // Handle the case where the node ID is not found in idTraitMapLocal
                    nodeGrid[i][j].trait3 = 0; // Or any default value you'd like to use
                    nodeGrid[i][j].trait3Values.clear();
                    nodeGrid[i][j].trait3Values.push_back(nodeGrid[i][j].trait3);
                }
            }
        }
        // Update the values of minTrait and maxTrait after modifying nodeGrid
        std::tie(minTrait, maxTrait) = findTraitBounds(nodeGrid);

    }


    if (option == 1) {
        // Prompt the user for the number of agents
        //int numAgents;
        //std::cout << "Enter the number of agents: ";
        //std::cin >> numAgents;

        // Initialize agents after processing the shapefile
        agents = initializeAgents(nodeGrid, numAgents, minTrait, maxTrait);
        // Find the agent with the maximum trait2 value
        auto maxTrait2It = std::max_element(agents.begin(), agents.end(),
            [](const Agent& a, const Agent& b) {
                return a.trait2 < b.trait2;  // Compare based on trait2
            });

        double maxTrait2 = (maxTrait2It != agents.end()) ? maxTrait2It->trait2 : std::numeric_limits<double>::lowest();
        //emit logMessage("Maior orientação dos agentes é: " + QString::number(maxTrait2) );
        //emit logMessage("Maior orientação dos lugares é: " + QString::number(maxTrait));
        if (maxTrait2 > maxTrait) {
            // Loop over the maxTrait value
            for (int i = 1; i <= maxTrait2; ++i) {
                // Get color for each value using getColorForValue function
                QColor color = getColorForValue(i, maxTrait2);
                // Add the color to the vector
                colorVector.push_back(color);
            }
            emit emitirtiposDeUso(maxTrait2, colorVector);
        }
        else {
            // Loop over the maxTrait value
            for (int i = 1; i <= maxTrait; ++i) {
                // Get color for each value using getColorForValue function
                QColor color = getColorForValue(i, maxTrait);
                // Add the color to the vector
                colorVector.push_back(color);
            }
            emit emitirtiposDeUso(maxTrait, colorVector);
        }
        // Run the simulation
        batelada(nodeGrid, agents, pathMap, minTrait, maxTrait, filenameToSave, numAgents, interactions, alpha, beta, pesoAgente, pesoLugar, steps, stepSize, gridHeatPath, gridHeatPathVector, AgentsVector);
        emit agentesIniciadosVector(AgentsVector);
        emit mandargridHeatPathVector(gridHeatPathVector);
        
        //emit exportarHeatPath(gridHeatPath);
    }

    else if (option == 0) {
        // Prompt the user for the number of agents
        //int numAgents;
        //std::cout << "Enter the number of agents: ";
        //std::cin >> numAgents;

        // Initialize agents after processing the shapefile
        agents = initializeAgents(nodeGrid, numAgents, minTrait, maxTrait);
        // Find the agent with the maximum trait2 value
        auto maxTrait2It = std::max_element(agents.begin(), agents.end(),
            [](const Agent& a, const Agent& b) {
                return a.trait2 < b.trait2;  // Compare based on trait2
            });

        double maxTrait2 = (maxTrait2It != agents.end()) ? maxTrait2It->trait2 : std::numeric_limits<double>::lowest();
        //emit logMessage("Maior orientação dos agentes é: " + QString::number(maxTrait2));
        //emit logMessage("Maior orientação dos lugares é: " + QString::number(maxTrait));
        if (maxTrait2 > maxTrait) {
            // Loop over the maxTrait value
            for (int i = 1; i <= maxTrait2; ++i) {
                // Get color for each value using getColorForValue function
                QColor color = getColorForValue(i, maxTrait2);
                // Add the color to the vector
                colorVector.push_back(color);
            }
            emit emitirtiposDeUso(maxTrait2, colorVector);
        }
        else {
            // Loop over the maxTrait value
            for (int i = 1; i <= maxTrait; ++i) {
                // Get color for each value using getColorForValue function
                QColor color = getColorForValue(i, maxTrait);
                // Add the color to the vector
                colorVector.push_back(color);
            }
            emit emitirtiposDeUso(maxTrait, colorVector);
        }
        // Run the simulation
        run(nodeGrid, agents, pathMap, minTrait, maxTrait, filenameToSave, numAgents, interactions, alpha, beta, pesoAgente, pesoLugar, poligonos, gridHeatPath, gridHeatPathVector);
        calculateNodeColors(gridHeatPath);
        emit agentesIniciados(agents);
        emit exportarHeatPath(gridHeatPath);
        
    }

}

int ABM::batelada(std::vector<std::vector<Node>>& nodeGrid, std::vector<Agent>& agents, pathMap& pathMap, double minTrait, double maxTrait, std::string filenameToSave, int numAgents,int numIterations,double alpha,double beta,double pesoAgente,double pesoLugar, int steps, double stepSize, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, std::vector<std::vector<std::vector<Agent>>>& AgentsVector) {
    // Number of iterations for the simulation
    //int numIterations = 2000;  // You can adjust this value
    //int steps = 0;

    // Contamination factor
    //double gamma = 0;  // You can adjust this value
    //double theta = 0;
    //double alpha = 0;
    //double beta = 0;
    //double pesoAgente = 0;
    //double pesoLugar = 0;
    //double stepSize = 0;

    std::vector<double> iterationNumbers;

    //createImageFromNodeGrid(nodeGrid, "Imagem.png");

    //std::cout << "Enter the number of iterations: ";
    //std::cin >> numIterations;

    //std::cout << "Enter the number of variations steps in the variables: ";
    //std::cin >> steps;

    //std::cout << "Enter the size of steps in the variables: ";
    //std::cin >> stepSize;

    //std::cout << "Enter the initial alpha value: ";
    //std::cin >> alpha;

    //std::cout << "Enter the initial beta value: ";
    //std::cin >> beta;

    //std::cout << "Enter the agent contamination factor: ";
    //std::cin >> pesoAgente;

    //std::cout << "Enter the place contamination factor: ";
    //std::cin >> pesoLugar;

    double Maiorcaminho = getHighestVisitedNodes(pathMap);

    // Width of the progress bar
    const int progressBarWidth = 50;

    //auto [minTrait, maxTrait] = findTraitBounds(nodeGrid);

    //std::cout << "Running simulation:" << std::endl;
    emit logMessage("Running simulation:");

    //std::ofstream BateladaPlotScript(filenameToSave + "_Ag_" + std::to_string(agents.size()) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_Batelada.plot");

    // Fill the 2D vector with entropy values
    std::vector<std::vector<std::tuple<double, double>>> entropyValues = fillEntropyValues( steps, steps, numIterations, nodeGrid, agents, pathMap, stepSize, alpha, beta, pesoAgente, pesoLugar, Maiorcaminho, filenameToSave, maxTrait, gridHeatPath, gridHeatPathVector, AgentsVector);

    // Create vectors to store the first and second values
    std::vector<std::vector<double>> firstValues(steps, std::vector<double>(steps));
    std::vector<std::vector<double>> secondValues(steps, std::vector<double>(steps));

    // Generate x and y labels
    std::vector<double> alphaLabels;
    std::vector<double> betaLabels;
    for (int i = 0; i < steps; ++i) {
        alphaLabels.push_back(alpha + i * stepSize);
        betaLabels.push_back(beta + i * stepSize);
    }

    // Iterate over the entropy values and extract the first and second values
    for (int i = 0; i < steps; ++i) {
        for (int j = 0; j < steps; ++j) {
            firstValues[i][j] = std::get<0>(entropyValues[i][j]);
            secondValues[i][j] = std::get<1>(entropyValues[i][j]);
        }
    }

    // Call plotHeatmap once for the first values
    //plotHeatmapGNU(BateladaPlotScript, firstValues, "Entropy_Heatmap_Agents", alphaLabels, betaLabels, filenameToSave, numAgents, numIterations, steps, int(stepSize * 100), int(alpha * 100), int(beta * 100), int(pesoAgente * 100), int(pesoLugar*100));

    
    // Call plotHeatmap once for the second values
    //plotHeatmapGNU(BateladaPlotScript, secondValues, "Entropy_Heatmap_Places", alphaLabels, betaLabels, filenameToSave, numAgents, numIterations, steps, int(stepSize * 100), int(alpha * 100), int(beta * 100), int(pesoAgente * 100), int(pesoLugar * 100));
    
    emit logMessage("Generating Graphs:");

    //QStringList arguments;
    //arguments << QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(agents.size()) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_Batelada.plot");

    //QThread thread;
    //thread.start();

    //QProcess gnuplotProcess;
    //gnuplotProcess.moveToThread(&thread);

    //gnuplotProcess.start("gnuplot", arguments);
    //gnuplotProcess.waitForFinished();

    //thread.quit();
    //thread.wait();

    //system(("gnuplot " + filenameToSave + "_Ag_" + std::to_string(agents.size()) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_Batelada.plot").c_str());

    //BateladaPlotScript.close();

    emit  messageBoxEnd();

    emit graphHeatMap(firstValues, secondValues);

    emit graphHeatMapQT(firstValues, "Agents Entropy 2D Heatmap", alphaLabels, betaLabels);
    emit graphHeatMapQT(secondValues, "Places Entropy 2D Heatmap", alphaLabels, betaLabels);
    emit criarMosaico();

    //createImageFromNodeGrid(nodeGrid, filenameToSave);

    //emit ImageProduced3(QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Steps_" + std::to_string(steps) + "_Stepsize_" + std::to_string(int(stepSize * 100)) + "_InitialAlpha_" + std::to_string(int(alpha * 100)) + "_InitialBeta_" + std::to_string(int(beta * 100)) + "Gamma_" + std::to_string(int(pesoAgente * 100)) + "_Theta_" + std::to_string(int(pesoLugar * 100)) + "_"));
    
    return 0;
}

void ABM::plotHeatmapGNU(std::ofstream& BateladaPlotScript, const std::vector<std::vector<double>>& heatmapData, const std::string& title, const std::vector<double>& xLabels, const std::vector<double>& yLabels, const std::string& filenameToSave, int numAgents, int numIterations, int steps, int stepSize, int alpha, int beta, int pesoAgente, int pesoLugar) {
    std::ofstream dataFile( filenameToSave + "heatmap_data_" + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Steps_" + std::to_string(steps) + "_Stepsize_" + std::to_string(stepSize) + "_InitialAlpha_" + std::to_string(alpha) + "_InitialBeta_" + std::to_string(beta) + "Gamma_" + std::to_string(pesoAgente) + "_Theta_" + std::to_string(pesoLugar) + "_" + title + ".dat");
    for (const auto& row : heatmapData) {
        for (double value : row) {
            dataFile << value << " ";
        }
        dataFile << std::endl;
    }
    dataFile.close();


    Gnuplot gp;
    gp << "set terminal pngcairo size 1000,846 enhanced font 'Arial,12'\n"; // Set PNG terminal with specified resolution
    gp << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Steps_" + std::to_string(steps) + "_Stepsize_" + std::to_string(stepSize) + "_InitialAlpha_" + std::to_string(alpha) + "_InitialBeta_" + std::to_string(beta) + "Gamma_" + std::to_string(pesoAgente) + "_Theta_" + std::to_string(pesoLugar) + "_" + title << ".png'\n"; // Set output filename
    gp << "set title '" << title << "'\n";
    gp << "set xlabel 'Alpha'\n";
    gp << "set ylabel 'Beta'\n";
    gp << "unset key\n";
    gp << "set view map\n";
    gp << "set palette rgb 33,13,10\n";
    gp << "set xtics (";
    for (size_t i = 0; i < xLabels.size(); ++i) {
        if (i != 0) gp << ", ";
        gp << "'" << std::fixed << std::setprecision(2) << xLabels[i] << "' " << i;
    }
    gp << ")\n";
    gp << "set ytics (";
    for (size_t i = 0; i < yLabels.size(); ++i) {
        if (i != 0) gp << ", ";
        gp << "'" << std::fixed << std::setprecision(2) << yLabels[i] << "' " << i;
    }
    gp << ")\n";
    gp << "plot '" + filenameToSave + "heatmap_data_" + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Steps_" + std::to_string(steps) + "_Stepsize_" + std::to_string(stepSize) + "_InitialAlpha_" + std::to_string(alpha) + "_InitialBeta_" + std::to_string(beta) + "Gamma_" + std::to_string(pesoAgente) + "_Theta_" + std::to_string(pesoLugar) + "_" + title + ".dat' matrix with image\n";
    gp.flush();
    //std::cin.get();  // Wait for user input to close the plot
}

// Fill a 2D vector with entropy values based on gamma and theta variations
std::vector<std::vector<std::tuple<double, double>>> ABM::fillEntropyValues(int alphaSteps, int betaSteps, int numIterations, std::vector<std::vector<Node>>& nodeGrid, std::vector<Agent>& agents, const pathMap& pathMap, double stepSize, double alpha, double beta, double pesoAgente, double pesoLugar, double Maiorcaminho, std::string filenameToSave, double maxTraitOut, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector, std::vector<std::vector<std::vector<Agent>>>& AgentsVector) {

    std::vector<std::vector<std::tuple<double, double>>> entropyValues(alphaSteps, std::vector<std::tuple<double, double>>(betaSteps));
    std::vector<std::vector<bool>> Batelada(alphaSteps, std::vector<bool>(betaSteps));
    std::vector<std::vector<std::vector<double>>> iterationNumbers(alphaSteps, std::vector<std::vector<double>>(betaSteps, std::vector<double>(numIterations)));
    std::vector<std::vector<std::vector<double>>> AGENTSentropyValues(alphaSteps, std::vector<std::vector<double>>(betaSteps, std::vector<double>(numIterations)));
    std::vector<std::vector<int>> numAgents(alphaSteps, std::vector<int>(betaSteps));
    std::vector<std::vector<std::vector<double>>> GRIDentropyValues(alphaSteps, std::vector<std::vector<double>>(betaSteps, std::vector<double>(numIterations)));
    std::vector<std::vector<double>> minTrait(alphaSteps, std::vector<double>(betaSteps));
    std::vector<std::vector<double>> maxTrait(alphaSteps, std::vector<double>(betaSteps));
    std::vector<std::vector<std::vector<std::vector<double>>>> agentTraitValuesContainer(alphaSteps, std::vector<std::vector<std::vector<double>>>(betaSteps, std::vector<std::vector<double>>(agents.size(), std::vector<double>(numIterations, 0.0))));
    std::vector<std::vector<std::vector<std::vector<double>>>> nodesTraitValuesContainer(alphaSteps, std::vector<std::vector<std::vector<double>>>(betaSteps, std::vector<std::vector<double>>(nodeGrid.size(), std::vector<double>(nodeGrid[0].size(), 0.0))));

    std::vector<std::vector<std::vector<std::pair<int, std::vector<QColor>>>>> idColorsPairsVector(alphaSteps, std::vector<std::vector<std::pair<int, std::vector<QColor>>>>(betaSteps));
    std::vector<std::vector<std::vector<std::pair<int, std::vector<double>>>>> traitsPoligonoVector(alphaSteps, std::vector<std::vector<std::pair<int, std::vector<double>>>>(betaSteps));

    int totalIterations = alphaSteps * betaSteps;

    // Create copies of nodeGrid and agents for this iteration
    std::vector<std::vector<Node>> nodeGridCopy = nodeGrid;
    std::vector<Agent> agentsCopy = agents;

    for (int i = 0; i < alphaSteps; ++i) {
        for (int j = 0; j < betaSteps; ++j) {

            nodeGridCopy = nodeGrid;
            agentsCopy = agents;

            double alpha = i * stepSize;
            double beta = j * stepSize;

            auto result = calculateEntropyAgentsBatelada(alpha, beta, numIterations, nodeGridCopy, agentsCopy, pathMap, stepSize, pesoAgente, pesoLugar, Maiorcaminho, filenameToSave, 0, idColorsPairsVector, gridHeatPath, gridHeatPathVector, i, j);

            entropyValues[i][j] = std::get<0>(result);
            Batelada[i][j] = std::get<1>(result);
            iterationNumbers[i][j] = std::get<2>(result);
            AGENTSentropyValues[i][j] = std::get<3>(result);
            numAgents[i][j] = std::get<4>(result);
            GRIDentropyValues[i][j] = std::get<5>(result);
            minTrait[i][j] = std::get<6>(result);
            maxTrait[i][j] = std::get<7>(result);
            agentTraitValuesContainer[i][j] = std::get<9>(result);
            nodesTraitValuesContainer[i][j] = std::get<8>(result);
            calculateNodeColors(gridHeatPath);
            gridHeatPathVector[i][j] = gridHeatPath;

            AgentsVector[i][j] = agentsCopy;

            std::unordered_map<int, std::vector<QColor>> idColorsMap;
            std::unordered_map<int, std::vector<double>> traitsPoligonoMap;
            std::unordered_map<int, size_t> maxTrait3ValuesLength;

            for (const auto& row : nodeGridCopy) {
                for (const auto& node : row) {
                    maxTrait3ValuesLength[node.id] = std::max(maxTrait3ValuesLength[node.id], node.trait3Values.size());
                }
            }

            for (const auto& row : nodeGridCopy) {
                for (const auto& node : row) {
                    if (node.trait3Values.size() < maxTrait3ValuesLength[node.id]) {
                        continue;
                    }
                    auto& idColors = idColorsMap[node.id];
                    auto& traitsPoligono = traitsPoligonoMap[node.id];
                    idColors.resize(numIterations);
                    traitsPoligono.resize(numIterations);

                    for (size_t k = 0; k < numIterations; ++k) {
                        if (node.trait3Values.empty()) {
                            idColors[k] = QColor(Qt::white);
                            traitsPoligono[k] = 0;
                        }
                        else if (k < node.trait3Values.size()) {
                            idColors[k] = getColorForValue(node.trait3Values[k], maxTraitOut);
                            traitsPoligono[k] = node.trait3Values[k];
                        }
                        else {
                            idColors[k] = idColors[k - 1];
                            traitsPoligono[k] = traitsPoligono[k - 1];
                        }
                    }
                }
            }

            std::vector<std::pair<int, std::vector<QColor>>> idColorsPairs;
            for (const auto& entry : idColorsMap) {
                idColorsPairs.emplace_back(entry.first, entry.second);
            }

            std::vector<std::pair<int, std::vector<double>>> traitsPoligono;
            for (const auto& entry : traitsPoligonoMap) {
                traitsPoligono.emplace_back(entry.first, entry.second);
            }

            idColorsPairsVector[i][j] = idColorsPairs;
            traitsPoligonoVector[i][j] = traitsPoligono;

            int completed = i * betaSteps + j + 1;
            double progress = static_cast<double>(completed) / totalIterations;
            emit progressUpdated(progress);
        }
    }

    emit colorsForTraitValuesEmittedBatelada(idColorsPairsVector, traitsPoligonoVector);
    emit salvarplotsinterno(Batelada, iterationNumbers, AGENTSentropyValues, numAgents, GRIDentropyValues, minTrait, maxTrait, agentTraitValuesContainer, nodesTraitValuesContainer);
    emit progressUpdated(1.0f); // Ensure 100% progress is reported at the end

    return entropyValues;
}


std::tuple <std::tuple<double, double>, bool, std::vector<double>, std::vector<double>, int, std::vector<double>, double, double, std::vector<std::vector<double>>, std::vector<std::vector<double>> > ABM::calculateEntropyAgentsBatelada( double alpha, double beta, int numIterations, std::vector<std::vector<Node>>& nodeGrid, std::vector<Agent>& agents,const pathMap& pathMap, double stepSize, double pesoAgente, double pesoLugar, double Maiorcaminho, std::string filenameToSave, int contador, std::vector<std::vector<std::vector<std::pair<int, std::vector<QColor>>>>> idColorsPairsVector, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector,int alphaSteps, int betaSteps) {
    std::vector<double> AGENTSentropyValues;
    std::vector<double> GRIDentropyValues;
    double PlacesentropyValuesSum = 0.0;
    double AGENTSentropyValuesSum = 0.0;
    std::unordered_set<double> uniqueTrait2Values;
    std::unordered_set<double> uniqueTrait3Values;
    std::vector<double> iterationNumbers;
    // Width of the progress bar

    auto [minTrait, maxTrait] = findTraitBounds(nodeGrid);
    for (int iteration = 0; iteration < numIterations; ++iteration) {
        // Update agent traits
        updateAgentTraits(agents, nodeGrid, pesoAgente, pesoLugar, minTrait, maxTrait, GRIDentropyValues, AGENTSentropyValues);

        // Move agents to the next access point
        moveAgents(agents, nodeGrid, pathMap, alpha, beta, Maiorcaminho, maxTrait, gridHeatPath, gridHeatPathVector, alphaSteps, betaSteps);

        //GRIDentropyValues.push_back(calculateShannonEntropyGRID(nodeGrid));
        //AGENTSentropyValues.push_back(calculateAgentShannonEntropy(agents));
        iterationNumbers.push_back(iteration);

        PlacesentropyValuesSum += GRIDentropyValues.back();
        AGENTSentropyValuesSum += AGENTSentropyValues.back();

    }

    double numIterationsDouble = static_cast<double>(numIterations);
    double PlacesentropyValuesAverage = PlacesentropyValuesSum / numIterationsDouble;
    double AGENTSentropyValuesAverage = AGENTSentropyValuesSum / numIterationsDouble;

    //emit logMessage("Entropia media Agentes alpha: " + QString::number(alphaSteps) + " beta: " + QString::number(betaSteps) + " entropia = "  + QString::number(AGENTSentropyValuesAverage) );
    //emit logMessage("Entropia media Lugares alpha: " + QString::number(alphaSteps) + " beta: " + QString::number(betaSteps) + " entropia = " + QString::number(PlacesentropyValuesAverage));
    
    int numAgents = agents.size();

    std::vector<std::vector<double>> agentTraitValues(agents.size(), std::vector<double>(iterationNumbers.size(), 0.0));

    // Write agents' trait data
    for (size_t j = 0; j < iterationNumbers.size(); ++j) {
        for (size_t i = 0; i < agents.size(); ++i) {
            agentTraitValues[i][j] = agents[i].trait2Values[j]; // Populate agentTraitValues
        }
    }

    std::vector<std::vector<double>> nodesTraitValues(nodeGrid.size());

    // Resize nodesTraitValues if necessary
    if (nodesTraitValues.size() != nodeGrid.size()) {
        nodesTraitValues.resize(nodeGrid.size());
    }
    for (size_t i = 0; i < nodesTraitValues.size(); ++i) {
        if (nodesTraitValues[i].size() - 1 != iterationNumbers.size()) {
            nodesTraitValues[i].resize(iterationNumbers.size());
        }
    }

    // Write nodes' trait data
    for (size_t j = 0; j < iterationNumbers.size(); ++j) {
        for (size_t i = 0; i < nodeGrid.size(); ++i) {
            for (size_t k = 0; k < nodeGrid[i].size(); ++k) {
                if (nodeGrid[i][k].trait3Values.size() > 1) {
                    size_t index = std::min(j, nodeGrid[i][k].trait3Values.size() - 1);
                    nodesTraitValues[i][j] = nodeGrid[i][k].trait3Values[index];
                }
            }
        }
    }

    
    ///Plotting part
    bool Batelada = true;
    
    //plotSingleGraphsGNU(BateladaPlotScript, Batelada, iterationNumbers, AGENTSentropyValues, filenameToSave, numAgents, numIterations, pesoLugar, pesoAgente, beta, alpha, GRIDentropyValues, nodeGrid, agents, contador, minTrait, maxTrait);

    ///End of plotting part
    return std::make_tuple(std::make_tuple(AGENTSentropyValuesAverage, PlacesentropyValuesAverage), Batelada, iterationNumbers, AGENTSentropyValues, numAgents, GRIDentropyValues, minTrait, maxTrait, nodesTraitValues, agentTraitValues);
}

// Main simulation loop with a progress bar
void ABM::run(std::vector<std::vector<Node>>& nodeGrid, std::vector<Agent>& agents, pathMap& pathMap, double minTrait, double maxTrait, std::string filenameToSave, int numAgents,int numIterations,double alpha,double beta,double pesoAgente,double pesoLugar, std::vector<PolygonData> poligonos, GridHeatPath& gridHeatPath, std::vector<std::vector<GridHeatPath>>& gridHeatPathVector) {

    // Number of iterations for the simulation
    //int numIterations = 0;  // You can adjust this value

    // Contamination factor
    //double alpha = 0;
    //double beta = 0;
    //double pesoAgente = 0;
    //double pesoLugar = 0;


    std::vector<double> GRIDentropyValues;
    std::vector<double> AGENTSentropyValues;
    std::vector<double> iterationNumbers;
    std::vector<double> AgentsTraits;
    std::vector<double> PlacesTraits;

    //createImageFromNodeGrid(nodeGrid, "Imagem.png");

    //std::cout << "Enter the number of iterations: ";
    //std::cin >> numIterations;

    //std::cout << "Enter the alpha value: ";
    //std::cin >> alpha;

    //std::cout << "Enter the beta value: ";
    //std::cin >> beta;

    //std::cout << "Enter the agent contamination factor: ";
    //std::cin >> pesoAgente;

    //std::cout << "Enter the place contamination factor: ";
    //std::cin >> pesoLugar;

    // Width of the progress bar
    const int progressBarWidth = 50;

    //auto [minTrait, maxTrait] = findTraitBounds(nodeGrid);

    double Maiorcaminho = getHighestVisitedNodes(pathMap);

    //std::cout << "Running simulation:" << std::endl;
    emit logMessage("Running simulation:");

    for (int iteration = 0; iteration < numIterations; ++iteration) {
        // Update agent traits
        updateAgentTraits(agents, nodeGrid, pesoAgente, pesoLugar, minTrait, maxTrait, GRIDentropyValues, AGENTSentropyValues);

        // Move agents to the next access point
        moveAgents(agents, nodeGrid, pathMap, alpha, beta, Maiorcaminho, maxTrait, gridHeatPath, gridHeatPathVector,0,0);

        //double GRIDentropy = calculateShannonEntropyGRID(nodeGrid);
        //GRIDentropyValues.push_back(GRIDentropy);
        //double AGENTentropy = calculateAgentShannonEntropy(agents);
        //AGENTSentropyValues.push_back(AGENTentropy);
        iterationNumbers.push_back(iteration);

        // Progress bar
        float progress = static_cast<float>(iteration + 1) / numIterations;
        int barWidth = static_cast<int>(progressBarWidth * progress);

        emit progressUpdated(progress);
        //std::cout << "\r[" << std::string(barWidth, '#') << std::string(progressBarWidth - barWidth, ' ') << "] "
        //    << std::fixed << std::setprecision(2) << progress * 100.0 << "%";
        //std::cout.flush();
    }
    int contador = 0;

    //std::ofstream SinglePlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_Single.plot");

    //plotSingleGraphsGNU( SinglePlotScript, false, iterationNumbers, AGENTSentropyValues, filenameToSave, numAgents, numIterations, pesoLugar, pesoAgente, beta, alpha, GRIDentropyValues, nodeGrid, agents, contador, minTrait, maxTrait);

    auto result = plotSingleGraphsQT( iterationNumbers,  AGENTSentropyValues, filenameToSave,  numAgents,  numIterations,  pesoLugar,  pesoAgente,  beta,  alpha, GRIDentropyValues, nodeGrid, agents,  contador,  minTrait,  maxTrait);
    //std::vector<std::vector<double>> agentTraitValues =  std::get<0>(result);
    //std::vector<std::vector<double>> nodesTraitValuesstd =  std::get<1>(result);
    
    emit logMessage("Generating Graphs:");

    //QStringList arguments;
    //arguments << QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(agents.size()) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_Single.plot");

    //QThread thread;
    //thread.start();

    //QProcess gnuplotProcess;
    //gnuplotProcess.moveToThread(&thread);

    //gnuplotProcess.start("gnuplot", arguments);
    //gnuplotProcess.waitForFinished();

    //thread.quit();
    //thread.wait();

    //system(("gnuplot " + filenameToSave + "_Ag_" + std::to_string(agents.size()) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_Single.plot").c_str());

    //SinglePlotScript.close();

    emit  messageBoxEnd();

    createImageFromNodeGrid(nodeGrid, filenameToSave);

    std::unordered_map<int, std::vector<QColor>> idColorsMap;
    std::unordered_map<int, std::vector<double>> traitsPoligonoMap;

    // Store the maximum trait3Values length for each ID
    std::unordered_map<int, size_t> maxTrait3ValuesLength;

    // First pass: Find the maximum trait3Values length for each ID
    for (const auto& row : nodeGrid) {
        for (const auto& node : row) {
            maxTrait3ValuesLength[node.id] = std::max(maxTrait3ValuesLength[node.id], node.trait3Values.size());
        }
    }

    // Second pass: Fill idColorsMap and traitsPoligonoMap with trait values of the node that has the longest trait3Values for each ID
    for (const auto& row : nodeGrid) {
        for (const auto& node : row) {
            if (node.trait3Values.size() != maxTrait3ValuesLength[node.id]) {



            }
            else {

                // Find or create an entry in the idColorsMap for this node's ID
                auto& idColors = idColorsMap[node.id];
                auto& traitsPoligono = traitsPoligonoMap[node.id];

                // Resize idColors and traitsPoligono to match numIterations
                idColors.resize(numIterations);
                traitsPoligono.resize(numIterations);

                // Fill idColors and traitsPoligono with trait values of the node that has the longest trait3Values
                for (size_t i = 0; i < numIterations; ++i) {
                    if (node.trait3Values.empty()) {
                        // If node.trait3Values is empty, set 0 and white
                        idColors[i] = QColor(Qt::white);
                        traitsPoligono[i] = 0;
                    }
                    else if (i < node.trait3Values.size()) {
                        // If node.trait3Values is shorter than numIterations, repeat the last value
                        idColors[i] = getColorForValue(node.trait3Values[i], maxTrait);
                        traitsPoligono[i] = node.trait3Values[i];
                    }
                    else {
                        // Repeat the last value
                        idColors[i] = idColors[i-1];
                        traitsPoligono[i] = traitsPoligono[i-1];
                    }
                }
            }
        }
    }

    // Convert idColorsMap to idColorsPairs
    std::vector<std::pair<int, std::vector<QColor>>> idColorsPairs;
    for (const auto& entry : idColorsMap) {
        idColorsPairs.emplace_back(entry.first, entry.second);
    }

    // Convert idColorsMap to idColorsPairs
    std::vector<std::pair<int, std::vector<double>>> traitsPoligono;
    for (const auto& entry : traitsPoligonoMap) {
        traitsPoligono.emplace_back(entry.first, entry.second);
    }

    //logMessage(QString::number(polygonsLocalCore.size()));
    // Emit the vector containing pairs of ID and corresponding colors
    emit colorsForTraitValuesEmitted(idColorsPairs, traitsPoligono);
    //emit agentesIniciados(agents);

    //emit ImageProduced2(QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_AgentEntropyOverTime.png"),
        //QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_PlaceEntropyOverTime.png"),
        //QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.png"),
        //QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agents_traits.png"),
        //QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_nodes_traits.png"));

    // Print a new line after completing the progress bar
    //std::cout << std::endl;
    
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> ABM::plotSingleGraphsQT(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait) {

    //std::ofstream agentEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agent_entropy.dat");
    //std::ofstream placeEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_place_entropy.dat");


    // Save agent entropy data
    // for (size_t i = 0; i < iterationNumbers.size(); ++i) {
    //     agentEntropyData << iterationNumbers[i] << " " << AGENTSentropyValues[i] << "\n";
    // }
    // agentEntropyData.close();

    // Save place entropy data
    // for (size_t i = 0; i < iterationNumbers.size(); ++i) {
    //     placeEntropyData << iterationNumbers[i] << " " << GRIDentropyValues[i] << "\n";
    // }
    // placeEntropyData.close();

    // Create combined data file for agent and place entropy
    // std::ofstream combinedEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.dat");

    // Save agent and place entropy data to the combined file
    // for (size_t i = 0; i < iterationNumbers.size(); ++i) {
    //     combinedEntropyData << iterationNumbers[i] << " " << AGENTSentropyValues[i] << " " << GRIDentropyValues[i] << "\n";
    // }
    // combinedEntropyData.close();


    //std::ofstream agentsTraitData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agents_traits.dat");

    std::vector<std::vector<double>> agentTraitValues(agents.size(), std::vector<double>(iterationNumbers.size(), 0.0));

    // Write header row for agents' trait data
    //agentsTraitData << "Iteration ";
    //for (size_t i = 0; i < agents.size(); ++i) {
    //    agentsTraitData << "Agent" << i + 1 << " ";

    //}
    //agentsTraitData << "\n";

    // Write agents' trait data
    for (size_t j = 0; j < iterationNumbers.size(); ++j) {
        //agentsTraitData << iterationNumbers[j] << " ";
        for (size_t i = 0; i < agents.size(); ++i) {
            //agentsTraitData << agents[i].trait2Values[j] << " ";
            agentTraitValues[i][j] = agents[i].trait2Values[j]; // Populate agentTraitValues
        }
        //agentsTraitData << "\n";
    }
    //agentsTraitData.close();

    //std::ofstream nodesTraitData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_nodes_traits.dat");

    std::vector<std::vector<double>> nodesTraitValues;

    // Write header row for nodes' trait data
    //nodesTraitData << "Iteration ";
    //for (size_t i = 0; i < nodeGrid.size(); ++i) {
    //    for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
            //nodesTraitData << "Node(" << i << "," << j << ") ";
    //    }
    //}
    //nodesTraitData << "\n";


    // Write nodes' trait data
    //for (size_t j = 0; j < iterationNumbers.size(); ++j) {
       // nodesTraitData << iterationNumbers[j] << " ";
    //    for (size_t i = 0; i < nodeGrid.size(); ++i) {
    //        for (size_t k = 0; k < nodeGrid[i].size(); ++k) {
    //            if (nodeGrid[i][k].trait3Values.size() > 1) {
    //                size_t index = std::min(j, nodeGrid[i][k].trait3Values.size() - 1);
                    //nodesTraitData << nodeGrid[i][k].trait3Values[index] << " ";
                    //nodesTraitValues[i][j] = nodeGrid[i][k].trait3Values[index];
    //            }
    //        }
    //    }
        //nodesTraitData << "\n";
    //}
    //nodesTraitData.close();

    for (size_t i = 0; i < nodeGrid.size(); ++i) {
        for (size_t k = 0; k < nodeGrid[i].size(); ++k) {
            if (nodeGrid[i][k].trait3Values.size() > 1) {
                nodesTraitValues.emplace_back(nodeGrid[i][k].trait3Values);
            }
        }
    }

    // Resize each column of nodesTraitValues that is smaller than iterationNumbers.size()
    for (size_t i = 0; i < nodesTraitValues.size(); ++i) {
        size_t colSize = nodesTraitValues[i].size();
        if (colSize < iterationNumbers.size()) {
            double lastValue = nodesTraitValues[i].empty() ? 0.0 : nodesTraitValues[i].back();
            nodesTraitValues[i].resize(iterationNumbers.size(), lastValue);
        }
    }

    double somaEntropiaAgentes = 0.0;
    double somaEntropiaLugares = 0.0;

    for (size_t i = 0; i < iterationNumbers.size(); ++i) {
        somaEntropiaAgentes += AGENTSentropyValues[i];
        somaEntropiaLugares += GRIDentropyValues[i];

    }
    double EntropiaMediaAgentes = somaEntropiaAgentes / AGENTSentropyValues.size();
    double EntropiaMediaLugares = somaEntropiaLugares / GRIDentropyValues.size();

    emit graphDataReady(iterationNumbers, AGENTSentropyValues, GRIDentropyValues, agentTraitValues,  minTrait,  maxTrait, nodesTraitValues);
    emit logMessage("Soma Entropia Agentes: " + QString::number(EntropiaMediaAgentes) + "Alpha: " + QString::number(alpha) + "Beta: " + QString::number(beta));
    emit logMessage("Soma Entropia Lugares: " + QString::number(EntropiaMediaLugares) + "Alpha: " + QString::number(alpha) + "Beta: " + QString::number(beta));
    return std::make_tuple(agentTraitValues, nodesTraitValues);
}

void ABM::plotSingleGraphsGNU(std::ofstream& ReceivedPlotScript, bool Batelada, std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait) {
    int decimalPlaces = 2;
    std::ofstream agentEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agent_entropy.dat");
    std::ofstream placeEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_place_entropy.dat");

    //std::cout << "Agents size: " << agents.size() << std::endl;
    //std::cout << "Nodes size: " << nodeGrid.size() << std::endl;

    // Save agent entropy data
    for (size_t i = 0; i < iterationNumbers.size(); ++i) {
        agentEntropyData << iterationNumbers[i] << " " << AGENTSentropyValues[i] << "\n";
    }
    agentEntropyData.close();

    // Save place entropy data
    for (size_t i = 0; i < iterationNumbers.size(); ++i) {
        placeEntropyData << iterationNumbers[i] << " " << GRIDentropyValues[i] << "\n";
    }
    placeEntropyData.close();


    // Gnuplot commands to plot agent entropy
    //std::ofstream agentPlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agent_entropy.plot");
    ReceivedPlotScript << "set terminal pngcairo size 850,850\n";
    ReceivedPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_AgentEntropyOverTime.png'\n";
    ReceivedPlotScript << "set title 'Entropy of the Agents vs. Iterations'\n";
    ReceivedPlotScript << "set xlabel 'Iterations'\n";
    ReceivedPlotScript << "set ylabel 'Entropy'\n";
    ReceivedPlotScript << "plot '" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_agent_entropy.dat' with lines title 'Entropy'\n";
    //ReceivedPlotScript.close();
    //system(("gnuplot " + filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agent_entropy.plot").c_str());

    emit progressUpdated(0.20);
    // Gnuplot commands to plot place entropy
    //std::ofstream placePlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_place_entropy.plot");
    ReceivedPlotScript << "set terminal pngcairo size 850,850\n";
    ReceivedPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_PlaceEntropyOverTime.png'\n";
    ReceivedPlotScript << "set title 'Entropy of the Places vs. Iterations'\n";
    ReceivedPlotScript << "set xlabel 'Iterations'\n";
    ReceivedPlotScript << "set ylabel 'Entropy'\n";
    ReceivedPlotScript << "plot '" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_place_entropy.dat' with lines title 'Entropy'\n";
    //ReceivedPlotScript.close();
    //system(("gnuplot " + filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_place_entropy.plot").c_str());

    emit progressUpdated(0.40);
    // Create combined data file for agent and place entropy
    std::ofstream combinedEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.dat");

    // Save agent and place entropy data to the combined file
    for (size_t i = 0; i < iterationNumbers.size(); ++i) {
        combinedEntropyData << iterationNumbers[i] << " " << AGENTSentropyValues[i] << " " << GRIDentropyValues[i] << "\n";
    }
    combinedEntropyData.close();

    // Gnuplot commands to plot agent and place entropy
    //std::ofstream combinedPlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.plot");
    ReceivedPlotScript << "set terminal pngcairo size 850,850\n";
    ReceivedPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_combined_entropy.png'\n";
    ReceivedPlotScript << "set title 'Entropy of Agents and Places vs. Iterations'\n";
    ReceivedPlotScript << "set xlabel 'Iterations'\n";
    ReceivedPlotScript << "set ylabel 'Entropy'\n";
    ReceivedPlotScript << "plot '" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_combined_entropy.dat' using 1:2 with lines title 'Agent Entropy', '' using 1:3 with lines title 'Place Entropy'\n";
    //ReceivedPlotScript.close();
    //system(("gnuplot " + filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.plot").c_str());
    
    emit progressUpdated(0.60);
    plotSingleGraphsGNU2(ReceivedPlotScript, Batelada, iterationNumbers, AGENTSentropyValues, filenameToSave, numAgents, numIterations, pesoLugar, pesoAgente, beta, alpha, GRIDentropyValues, nodeGrid, agents, contador, minTrait, maxTrait);

}

void ABM::plotSingleGraphsGNU2(std::ofstream& ReceivedPlotScript, bool Batelada, std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait) {
    std::ofstream agentsTraitData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agents_traits.dat");

    std::vector<std::vector<double>> agentTraitValues(agents.size(), std::vector<double>(iterationNumbers.size(), 0.0));

    // Write header row for agents' trait data
    agentsTraitData << "Iteration ";
    for (size_t i = 0; i < agents.size(); ++i) {
        agentsTraitData << "Agent" << i + 1 << " ";

    }
    agentsTraitData << "\n";

    // Write agents' trait data
    for (size_t j = 0; j < iterationNumbers.size(); ++j) {
        agentsTraitData << iterationNumbers[j] << " ";
        for (size_t i = 0; i < agents.size(); ++i) {
            agentsTraitData << agents[i].trait2Values[j] << " ";
            agentTraitValues[i][j] = agents[i].trait2Values[j]; // Populate agentTraitValues
        }
        agentsTraitData << "\n";
    }
    agentsTraitData.close();



    // Gnuplot commands to plot agents' traits over interactions
    //std::ofstream agentsPlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agents_traits.plot");
    ReceivedPlotScript << "set terminal pngcairo size 850,850\n";
    ReceivedPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_agents_traits.png'\n";
    ReceivedPlotScript << "set title 'All Agents Traits vs. Iterations'\n";
    ReceivedPlotScript << "set xlabel 'Iterations'\n";
    ReceivedPlotScript << "set ylabel 'Trait Values'\n";
    ReceivedPlotScript << "plot '" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_agents_traits.dat' using 1:2 with lines notitle";
    for (size_t i = 1; i < agents.size(); ++i) {
        ReceivedPlotScript << ", '' using 1:" << i + 2 << " with line notitle";
    }
    ReceivedPlotScript << "\n";
    //ReceivedPlotScript.close();
    //system(("gnuplot " + filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agents_traits.plot").c_str());

    emit progressUpdated(0.80);

    plotSingleGraphsGNU3(ReceivedPlotScript, Batelada, iterationNumbers, AGENTSentropyValues, filenameToSave, numAgents, numIterations, pesoLugar, pesoAgente, beta, alpha, GRIDentropyValues, nodeGrid, agents, contador, minTrait, maxTrait, agentTraitValues);

}

void ABM::plotSingleGraphsGNU3(std::ofstream& ReceivedPlotScript, bool Batelada, std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait, std::vector<std::vector<double>> agentTraitValues) {
    std::ofstream nodesTraitData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_nodes_traits.dat");

    std::vector<std::vector<double>> nodesTraitValues(nodeGrid.size());

    // Write header row for nodes' trait data
    nodesTraitData << "Iteration ";
    for (size_t i = 0; i < nodeGrid.size(); ++i) {
        for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
            nodesTraitData << "Node(" << i << "," << j << ") ";
        }
    }
    nodesTraitData << "\n";

    // Resize nodesTraitValues if necessary
    if (nodesTraitValues.size() != nodeGrid.size()) {
        nodesTraitValues.resize(nodeGrid.size());
    }
    for (size_t i = 0; i < nodesTraitValues.size(); ++i) {
        if (nodesTraitValues[i].size()-1 != iterationNumbers.size()) {
            nodesTraitValues[i].resize(iterationNumbers.size());
        }
    }

    // Write nodes' trait data
    for (size_t j = 0; j < iterationNumbers.size(); ++j) {
        nodesTraitData << iterationNumbers[j] << " ";
        for (size_t i = 0; i < nodeGrid.size(); ++i) {
            for (size_t k = 0; k < nodeGrid[i].size(); ++k) {
                if (nodeGrid[i][k].trait3Values.size() > 1) {
                    size_t index = std::min(j, nodeGrid[i][k].trait3Values.size() - 1);
                    nodesTraitData << nodeGrid[i][k].trait3Values[index] << " ";
                    nodesTraitValues[i][j] = nodeGrid[i][k].trait3Values[index];
                }
            }
        }
        nodesTraitData << "\n";
    }
    nodesTraitData.close();


    // Gnuplot commands to plot nodes' traits over interactions
    //std::ofstream nodesPlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_nodes_traits.plot");
    ReceivedPlotScript << "set terminal pngcairo size 850,850\n";
    ReceivedPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_nodes_traits.png'\n";
    ReceivedPlotScript << "set title 'All Nodes Traits vs. Iterations'\n";
    ReceivedPlotScript << "set xlabel 'Iterations'\n";
    ReceivedPlotScript << "set ylabel 'Trait Values'\n";
    ReceivedPlotScript << "plot ";

    size_t count = 0;
    for (size_t i = 0; i < nodeGrid.size(); ++i) {
        for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
            if (nodeGrid[i][j].trait3Values.size() > 1) {
                if (count != 0) {
                    ReceivedPlotScript << ", ";
                }
                ReceivedPlotScript << "'" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_nodes_traits.dat' using 1:" << count + 2 << " with lines notitle";
                count++;
            }
        }
    }
    ReceivedPlotScript << "\n";
    //ReceivedPlotScript.close();
    //system(("gnuplot " + filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_nodes_traits.plot").c_str());

    emit progressUpdated(1.0);

}

void ABM::plotSingleGraphsGNUCMD(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait) {
    int decimalPlaces = 2;
    std::ofstream agentEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agent_entropy.dat");
    std::ofstream placeEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_place_entropy.dat");

    //std::cout << "Agents size: " << agents.size() << std::endl;
    //std::cout << "Nodes size: " << nodeGrid.size() << std::endl;

    // Save agent entropy data
    for (size_t i = 0; i < iterationNumbers.size(); ++i) {
        agentEntropyData << iterationNumbers[i] << " " << AGENTSentropyValues[i] << "\n";
    }
    agentEntropyData.close();

    // Save place entropy data
    for (size_t i = 0; i < iterationNumbers.size(); ++i) {
        placeEntropyData << iterationNumbers[i] << " " << GRIDentropyValues[i] << "\n";
    }
    placeEntropyData.close();

    emit logMessage("Generating Graphs:");
    

    // Gnuplot commands to plot agent entropy
    std::ofstream agentPlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agent_entropy.plot");
    agentPlotScript << "set terminal pngcairo size 850,850\n";
    agentPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_AgentEntropyOverTime.png'\n";
    agentPlotScript << "set title 'Entropy of the Agents vs. Iterations'\n";
    agentPlotScript << "set xlabel 'Iterations'\n";
    agentPlotScript << "set ylabel 'Entropy'\n";
    agentPlotScript << "plot '" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_agent_entropy.dat' with lines title 'Entropy'\n";
    agentPlotScript.close();

    // Run gnuplot script using QProcess
    QStringList arguments;
    arguments << QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agent_entropy.plot");

    QThread thread;
    thread.start();

    QProcess gnuplotProcess;
    gnuplotProcess.moveToThread(&thread);

    gnuplotProcess.start("gnuplot", arguments);
    gnuplotProcess.waitForFinished();

    thread.quit();
    thread.wait();

    emit progressUpdated(0.20);

    // Gnuplot commands to plot place entropy
    std::ofstream placePlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_place_entropy.plot");
    placePlotScript << "set terminal pngcairo size 850,850\n";
    placePlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_PlaceEntropyOverTime.png'\n";
    placePlotScript << "set title 'Entropy of the Places vs. Iterations'\n";
    placePlotScript << "set xlabel 'Iterations'\n";
    placePlotScript << "set ylabel 'Entropy'\n";
    placePlotScript << "plot '" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_place_entropy.dat' with lines title 'Entropy'\n";
    placePlotScript.close();

    // Run gnuplot script using QProcess
    QStringList arguments2;
    arguments2 << QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_place_entropy.plot");

    QThread thread2;
    thread2.start();

    QProcess gnuplotProcess2;
    gnuplotProcess2.moveToThread(&thread2);

    gnuplotProcess2.start("gnuplot", arguments2);
    gnuplotProcess2.waitForFinished();

    thread2.quit();
    thread2.wait();

    emit progressUpdated(0.40);
    // Create combined data file for agent and place entropy
    std::ofstream combinedEntropyData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.dat");

    // Save agent and place entropy data to the combined file
    for (size_t i = 0; i < iterationNumbers.size(); ++i) {
        combinedEntropyData << iterationNumbers[i] << " " << AGENTSentropyValues[i] << " " << GRIDentropyValues[i] << "\n";
    }
    combinedEntropyData.close();

    // Gnuplot commands to plot agent and place entropy
    std::ofstream combinedPlotScript(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.plot");
    combinedPlotScript << "set terminal pngcairo size 850,850\n";
    combinedPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_combined_entropy.png'\n";
    combinedPlotScript << "set title 'Entropy of Agents and Places vs. Iterations'\n";
    combinedPlotScript << "set xlabel 'Iterations'\n";
    combinedPlotScript << "set ylabel 'Entropy'\n";
    combinedPlotScript << "plot '" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_combined_entropy.dat' using 1:2 with lines title 'Agent Entropy', '' using 1:3 with lines title 'Place Entropy'\n";
    combinedPlotScript.close();

    // Run gnuplot script using QProcess
    QStringList arguments3;
    arguments3 << QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.plot");

    QThread thread3;
    thread3.start();

    QProcess gnuplotProcess3;
    gnuplotProcess3.moveToThread(&thread3);

    gnuplotProcess3.start("gnuplot", arguments3);
    gnuplotProcess3.waitForFinished();

    thread3.quit();
    thread3.wait();

    emit progressUpdated(0.60);

    plotSingleGraphsGNU2CMD(iterationNumbers, AGENTSentropyValues, filenameToSave, numAgents, numIterations, pesoLugar, pesoAgente, beta, alpha, GRIDentropyValues, nodeGrid, agents, contador, minTrait, maxTrait);

}

void ABM::plotSingleGraphsGNU2CMD(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait) {
    std::ofstream agentsTraitData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agents_traits.dat");


    std::vector<std::vector<double>> agentTraitValues(agents.size(), std::vector<double>(iterationNumbers.size(), 0.0));

    // Write header row for agents' trait data
    agentsTraitData << "Iteration ";
    for (size_t i = 0; i < agents.size(); ++i) {
        agentsTraitData << "Agent" << i + 1 << " ";

    }
    agentsTraitData << "\n";

    // Write agents' trait data
    for (size_t j = 0; j < iterationNumbers.size(); ++j) {
        agentsTraitData << iterationNumbers[j] << " ";
        for (size_t i = 0; i < agents.size(); ++i) {
            agentsTraitData << agents[i].trait2Values[j] << " ";
            agentTraitValues[i][j] = agents[i].trait2Values[j]; // Populate agentTraitValues
        }
        agentsTraitData << "\n";
    }
    agentsTraitData.close();



    // Save the gnuplot script to a file for agents' traits
    QString agentsGnuplotScriptFilename = QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agents_traits.plot");
    std::ofstream agentsPlotScript(agentsGnuplotScriptFilename.toStdString());
    agentsPlotScript << "set terminal pngcairo size 850,850\n";
    agentsPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_agents_traits.png'\n";
    agentsPlotScript << "set title 'All Agents Traits vs. Iterations'\n";
    agentsPlotScript << "set xlabel 'Iterations'\n";
    agentsPlotScript << "set ylabel 'Trait Values'\n";
    agentsPlotScript << "plot '" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_agents_traits.dat' using 1:2 with lines notitle";
    for (size_t i = 1; i < agents.size(); ++i) {
        agentsPlotScript << ", '' using 1:" << i + 2 << " with line notitle";
    }
    agentsPlotScript << "\n";
    agentsPlotScript.close();

    // Run gnuplot script for agents' traits using QProcess
    QStringList agentsArguments;
    agentsArguments << agentsGnuplotScriptFilename;

    //QThread thread4;
    //thread4.start();

    QProcess agentsGnuplotProcess;
    //agentsGnuplotProcess.moveToThread(&thread4);

    agentsGnuplotProcess.start("gnuplot", agentsArguments);
    agentsGnuplotProcess.waitForFinished();

    //thread4.quit();
    //thread4.wait();

    emit progressUpdated(0.80);

    plotSingleGraphsGNU3CMD(iterationNumbers, AGENTSentropyValues, filenameToSave, numAgents, numIterations, pesoLugar, pesoAgente, beta, alpha, GRIDentropyValues, nodeGrid, agents, contador, minTrait, maxTrait, agentTraitValues);

}

void ABM::plotSingleGraphsGNU3CMD(std::vector<double> iterationNumbers, std::vector<double> AGENTSentropyValues, std::string filenameToSave, int numAgents, int numIterations, double pesoLugar, double pesoAgente, double beta, double alpha, std::vector<double> GRIDentropyValues, const std::vector<std::vector<Node>>& nodeGrid, const std::vector<Agent>& agents, int contador, double minTrait, double maxTrait, std::vector<std::vector<double>> agentTraitValues) {
    std::ofstream nodesTraitData(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_nodes_traits.dat");

    std::vector<std::vector<double>> nodesTraitValues(nodeGrid.size());

    // Write header row for nodes' trait data
    nodesTraitData << "Iteration ";
    for (size_t i = 0; i < nodeGrid.size(); ++i) {
        for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
            nodesTraitData << "Node(" << i << "," << j << ") ";
        }
    }
    nodesTraitData << "\n";

    // Resize nodesTraitValues if necessary
    if (nodesTraitValues.size() != nodeGrid.size()) {
        nodesTraitValues.resize(nodeGrid.size());
    }
    for (size_t i = 0; i < nodesTraitValues.size(); ++i) {
        if (nodesTraitValues[i].size() - 1 != iterationNumbers.size()) {
            nodesTraitValues[i].resize(iterationNumbers.size());
        }
    }

    // Write nodes' trait data
    for (size_t j = 0; j < iterationNumbers.size(); ++j) {
        nodesTraitData << iterationNumbers[j] << " ";
        for (size_t i = 0; i < nodeGrid.size(); ++i) {
            for (size_t k = 0; k < nodeGrid[i].size(); ++k) {
                if (nodeGrid[i][k].trait3Values.size() > 1) {
                    size_t index = std::min(j, nodeGrid[i][k].trait3Values.size() - 1);
                    nodesTraitData << nodeGrid[i][k].trait3Values[index] << " ";
                    nodesTraitValues[i][j] = nodeGrid[i][k].trait3Values[index];
                }
            }
        }
        nodesTraitData << "\n";
    }
    nodesTraitData.close();


    // Save the gnuplot script to a file
    QString gnuplotScriptFilename = QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_nodes_traits.plot");
    std::ofstream nodesPlotScript(gnuplotScriptFilename.toStdString());
    nodesPlotScript << "set terminal pngcairo size 850,850\n";
    nodesPlotScript << "set output '" << filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) << "_nodes_traits.png'\n";
    nodesPlotScript << "set title 'All Nodes Traits vs. Iterations'\n";
    nodesPlotScript << "set xlabel 'Iterations'\n";
    nodesPlotScript << "set ylabel 'Trait Values'\n";
    nodesPlotScript << "plot ";

    size_t count = 0;
    for (size_t i = 0; i < nodeGrid.size(); ++i) {
        for (size_t j = 0; j < nodeGrid[i].size(); ++j) {
            if (nodeGrid[i][j].trait3Values.size() > 1) {
                if (count != 0) {
                    nodesPlotScript << ", ";
                }
                nodesPlotScript << "'" << filenameToSave << "_Ag_" << numAgents << "_It_" << numIterations << "_Alpha_" << int(alpha * 100) << "_Beta_" << int(beta * 100) << "_PesoAgente_" << int(pesoAgente * 100) << "_PesoLugar_" << int(pesoLugar * 100) << "_nodes_traits.dat' using 1:" << count + 2 << " with lines notitle";
                count++;
            }
        }
    }
    nodesPlotScript << "\n";
    nodesPlotScript.close();

    // Run gnuplot script using QProcess
    QStringList arguments4;
    arguments4 << gnuplotScriptFilename;

    //QThread thread5;
    //thread5.start();

    QProcess gnuplotProcess4;
    //gnuplotProcess4.moveToThread(&thread5);

    gnuplotProcess4.start("gnuplot", arguments4);
    gnuplotProcess4.waitForFinished();

    //thread5.quit();
    //thread5.wait();

    emit progressUpdated(1);

    emit  messageBoxEnd();

    emit ImageProduced2(QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_AgentEntropyOverTime.png"),
        QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_PlaceEntropyOverTime.png"),
        QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_combined_entropy.png"),
        QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_agents_traits.png"),
        QString::fromStdString(filenameToSave + "_Ag_" + std::to_string(numAgents) + "_It_" + std::to_string(numIterations) + "_Alpha_" + std::to_string(int(alpha * 100)) + "_Beta_" + std::to_string(int(beta * 100)) + "_PesoAgente_" + std::to_string(int(pesoAgente * 100)) + "_PesoLugar_" + std::to_string(int(pesoLugar * 100)) + "_nodes_traits.png"));

}

void ABM::carregarPoligonos(const std::string& shapefilePath) {
    std::string shapefilePathNoExt;

    // Find the position of the last '.' character
    size_t lastDotPos2 = shapefilePath.find_last_of(".");
    if (lastDotPos2 == std::string::npos) {
        // No '.' character found, handle the error or return empty string as needed
        std::cerr << "No '.' character found in the provided shapefile path: " << shapefilePath << std::endl;
        shapefilePathNoExt = "";
    }
    else {
        // Extract the substring from the beginning of shapefilePath up to (but not including) the last '.'
        shapefilePathNoExt = shapefilePath.substr(0, lastDotPos2);
    }

    // Load polygons from the .polygons file
    std::vector<PolygonData> poligonos = loadPolygonsFromFile(shapefilePathNoExt + ".polygons");

    // Emit the polygonsReady signal with the loaded polygons
    emit polygonsReady(poligonos);
}

void ABM::inicio( std::string shapefilePath, std::string CAMINHOPASTA, int recSize, int option, int numAgents,int interactions,double alpha,double beta,double pesoAgente,double pesoLugar, int steps, double stepSize, int VoltarPraCasa, int DistribuicaoUsos, int Maxtraits, int MaxtraitsAgents, int DistribuicaoAgentes, int threads, int metodoEscolha, std::unordered_map<int, int> idTraitMap, std::vector<std::pair<int, int>> agentTraits) {
    pathMap pathMap;
    VoltarPraCasaLocal = VoltarPraCasa;
    DistribuicaoUsosLocal = DistribuicaoUsos;
    MaxtraitsLocal = Maxtraits;
    MaxtraitsAgentsLocal = MaxtraitsAgents;
    DistribuicaoAgentesLocal = DistribuicaoAgentes;
    threadsLocal = threads;
    metodoEscolhaLocal = metodoEscolha;
    idTraitMapLocal = idTraitMap;
    agentTraitsLocal = agentTraits;

    //std::cout << "Choose an option:" << std::endl;
    //std::cout << "1. Process shapefile and create nodeGrid" << std::endl;
    //std::cout << "2. Load existing nodeGrid from file" << std::endl;
    //std::cout << "option selected: " << option << " path: " << shapefilePath << std::endl;

    //std::cout << "Enter the number of rows: ";
    //std::cin >> numRows;
    //std::cout << "Enter the number of columns: ";
    //std::cin >> numCols;

    // Prompt the user for the filename to load the nodeGrid
    //std::string filenameToLoad;
    //std::cout << "Enter the filename to load the nodeGrid: ";
    //std::cin >> filenameToLoad;

    std::vector<std::vector<Node>> nodeGrid = loadNodeGrid(shapefilePath);

    //exportImageFromNodeGrid(nodeGrid);

    std::string shapefilePathNoExt;
    // Find the position of the last '.' character
    size_t lastDotPos2 = shapefilePath.find_last_of(".");
    if (lastDotPos2 == std::string::npos) {
        // No '.' character found, handle the error or return empty string as needed
        std::cerr << "No '.' character found." << std::endl;
        shapefilePathNoExt = "";
    }
    else {
        // Extract the substring from the beginning of fullPathStr up to (but not including) the last '.'
        shapefilePathNoExt = shapefilePath.substr(0, lastDotPos2);
    }

    pathMap = loadPaths(shapefilePathNoExt + ".paths");

    std::vector<PolygonData> poligonos = loadPolygonsFromFile(shapefilePathNoExt + ".polygons");

    emit polygonsReady(poligonos);

    std::string filenameToSave2;
    // Find the position of the last '\' character
    size_t lastSlashPos2 = shapefilePath.find_last_of("/");
    if (lastSlashPos2 == std::string::npos) {
        // No '\' character found, return empty string or fullPath as per requirement
        std::cerr << "No '/' character found." << std::endl;
        filenameToSave2 = "";
    }

    // Find the position of the last '.' character
    size_t lastDotPos3 = shapefilePath.find_last_of(".");
    if (lastDotPos3 == std::string::npos || lastDotPos3 < lastSlashPos2) {
        // No '.' character found after the last '\', or '.' character is before '\', return empty string
        std::cerr << "No '.' character found after the last '/', or '.' character is before '/'." << std::endl;
        filenameToSave2 = "";
    }

    // Extract the substring between the last '\' and the last '.'
    filenameToSave2 = shapefilePath.substr(lastSlashPos2 + 1, lastDotPos3 - lastSlashPos2 - 1);

    //savePathsToCSV(filenameToLoad + "_Paths.csv", pathMap);
    // Print the loaded nodeGrid
    //printNodeGrid(nodeGrid, numRows, numCols);
    chooseRunMethod(nodeGrid, pathMap, CAMINHOPASTA + "/" + filenameToSave2, option, numAgents, interactions, alpha, beta, pesoAgente, pesoLugar, steps, stepSize, poligonos);

    //std::cout << "Processing completed successfully." << std::endl;

    emit logMessage("Processing completed successfully.");
    //emit ImageProduced(QString::fromStdString(CAMINHOPASTA + "/" + filenameToSave2));


}

// Function to save polygons to a file
void ABM::savePolygonsToFile(const std::vector<PolygonData>& polygons, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Write the number of polygons to the file
        size_t numPolygons = polygons.size();
        file.write(reinterpret_cast<const char*>(&numPolygons), sizeof(numPolygons));

        // Write each polygon to the file
        for (const PolygonData& polygon : polygons) {
            // Write the ID of the polygon
            file.write(reinterpret_cast<const char*>(&polygon.id), sizeof(polygon.id));

            // Write the color of the polygon
            uint colorValue = polygon.color.rgb();
            file.write(reinterpret_cast<const char*>(&colorValue), sizeof(colorValue));

            // Write the number of points in the polygon
            size_t numPoints = polygon.points.size();
            file.write(reinterpret_cast<const char*>(&numPoints), sizeof(numPoints));

            // Write each point of the polygon
            for (const QPointF& point : polygon.points) {
                file.write(reinterpret_cast<const char*>(&point), sizeof(point));
            }
        }

        file.close();
    }
    else {
        // Failed to open the file
        std::cerr << "Error: Unable to open file for writing.\n";
    }
}

// Function to load polygons from a file
std::vector<PolygonData> ABM::loadPolygonsFromFile(const std::string& filename) {
    std::vector<PolygonData> polygons;
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        // Read the number of polygons from the file
        size_t numPolygons;
        file.read(reinterpret_cast<char*>(&numPolygons), sizeof(numPolygons));

        // Read each polygon from the file
        for (size_t i = 0; i < numPolygons; ++i) {
            PolygonData polygon;

            // Read the ID of the polygon
            file.read(reinterpret_cast<char*>(&polygon.id), sizeof(polygon.id));

            // Read the color of the polygon
            uint colorValue;
            file.read(reinterpret_cast<char*>(&colorValue), sizeof(colorValue));
            polygon.color = QColor::fromRgb(colorValue);

            // Read the number of points in the polygon
            size_t numPoints;
            file.read(reinterpret_cast<char*>(&numPoints), sizeof(numPoints));

            // Read each point of the polygon
            for (size_t j = 0; j < numPoints; ++j) {
                QPointF point;
                file.read(reinterpret_cast<char*>(&point), sizeof(point));
                polygon.points.push_back(point);
            }

            polygons.push_back(polygon);
        }

        file.close();
    }
    else {
        // Failed to open the file
        std::cerr << "Error: Unable to open file for reading.\n";
    }

    return polygons;
}