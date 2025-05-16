#!/bin/bash
# filepath: /Users/pedropacheco/Projects/dev/graphrag-agent/run.sh

# Define colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display banner
echo -e "${GREEN}=======================================${NC}"
echo -e "${YELLOW}       GraphRAG Flow Generator        ${NC}"
echo -e "${GREEN}=======================================${NC}"

# Function to display usage information
show_usage() {
    echo -e "\n${YELLOW}Usage:${NC}"
    echo -e "  ./run.sh [single_turn|multi_turn] [query]"
    echo -e "\n${YELLOW}Examples:${NC}"
    echo -e "  ./run.sh multi_turn                     # Start interactive multi-turn session"
    echo -e "  ./run.sh single_turn \"Create a RAG flow using Weaviate\" # Generate a flow in single-turn mode"
    echo
}

# Check if we have the required arguments
if [ $# -lt 1 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    show_usage
    exit 1
fi

# Get the mode argument
MODE=$1

# Validate mode
if [ "$MODE" != "single_turn" ] && [ "$MODE" != "multi_turn" ]; then
    echo -e "${RED}Error: Mode must be either 'single_turn' or 'multi_turn'${NC}"
    show_usage
    exit 1
fi

# Check if we need a query for single_turn mode
if [ "$MODE" = "single_turn" ]; then
    if [ $# -lt 2 ]; then
        echo -e "${RED}Error: single_turn mode requires a query${NC}"
        show_usage
        exit 1
    fi
    
    # Get the query (all remaining arguments)
    QUERY="${@:2}"
    
    echo -e "${GREEN}Running in single_turn mode with query:${NC} $QUERY"
    python src/graphrag_agent/flow_chat_v2.py single_turn --query "$QUERY"
else
    echo -e "${GREEN}Starting interactive multi_turn session...${NC}"
    python src/graphrag_agent/flow_chat_v2.py multi_turn
fi