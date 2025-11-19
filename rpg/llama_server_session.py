import requests
import json
from typing import Optional, Dict, List, Any


class LlamaServerSession:
    """Context manager for interacting with a local llama-server instance.
    
    Provides methods for sending prompts, managing conversation history,
    regenerating responses, and monitoring context usage.
    
    Usage:
        with LlamaServerSession("http://localhost:8080") as session:
            response = session.prompt("Hello, world!")
            print(response)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model: str = "default",
        temperature: float = 0.8,
        max_tokens: int = -1,
        top_p: float = 0.9,
        top_k: int = 40,
        seed: Optional[int] = None,
        **kwargs
    ):
        """Initialize the llama-server session.
        
        Args:
            base_url: URL of the llama-server instance
            model: Model identifier (usually "default" for llama-server)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate (-1 for unlimited)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
            seed: Random seed for reproducibility
            **kwargs: Additional llama.cpp parameters (min_p, repeat_penalty, etc.)
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.extra_params = kwargs
        
        # Conversation state
        self._messages: List[Dict[str, str]] = []
        self._system_prompt: Optional[str] = None
        self._ctx_messages: List[Dict[str, str]] = []  # Messages without regenerated ones
        
        # Session management
        self._session = requests.Session()
        self._is_connected = False
    
    def __enter__(self):
        """Enter the context manager."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.close()
        return False
    
    def connect(self) -> bool:
        """Verify connection to llama-server.
        
        Returns:
            True if connection successful, raises exception otherwise
        """
        try:
            # Try /health endpoint first
            response = self._session.get(f"{self.base_url}/health", timeout=5)
            self._is_connected = response.status_code == 200
        except requests.exceptions.RequestException:
            # Fall back to /props if /health not available
            try:
                response = self._session.get(f"{self.base_url}/props", timeout=5)
                self._is_connected = response.status_code == 200
            except requests.exceptions.RequestException as e:
                raise ConnectionError(
                    f"Failed to connect to llama-server at {self.base_url}: {e}"
                )
        
        if not self._is_connected:
            raise ConnectionError(f"llama-server returned status {response.status_code}")
        
        return True
    
    def close(self):
        """Close the session."""
        self._session.close()
        self._is_connected = False
    
    def system_prompt(self, new_prompt: Optional[str] = None) -> Optional[str]:
        """Get or set the system prompt.
        
        Args:
            new_prompt: If provided, sets a new system prompt
            
        Returns:
            The current system prompt
        """
        if new_prompt is not None:
            self._system_prompt = new_prompt
            # Update or add system message at the beginning
            if self._messages and self._messages[0].get("role") == "system":
                self._messages[0]["content"] = new_prompt
                if self._ctx_messages and self._ctx_messages[0].get("role") == "system":
                    self._ctx_messages[0]["content"] = new_prompt
            else:
                system_msg = {"role": "system", "content": new_prompt}
                self._messages.insert(0, system_msg)
                self._ctx_messages.insert(0, system_msg)
        
        return self._system_prompt
    
    def prompt(
        self,
        user_prompt: str,
        stream: bool = False,
        **override_params
    ) -> str:
        """Send a prompt and get a response.
        
        Args:
            user_prompt: The user's message
            stream: Whether to stream the response (not implemented in this version)
            **override_params: Override default generation parameters
            
        Returns:
            The assistant's response text
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to llama-server. Use with statement or call connect().")
        
        # Add user message to history
        user_msg = {"role": "user", "content": user_prompt}
        self._messages.append(user_msg)
        self._ctx_messages.append(user_msg)
        
        # Build request
        messages = self._messages.copy()
        
        # Merge parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        
        if self.seed is not None:
            params["seed"] = self.seed
        
        # Add extra params
        params.update(self.extra_params)
        params.update(override_params)
        
        # Make request
        response = self._session.post(
            f"{self.base_url}/v1/chat/completions",
            json=params,
            timeout=300
        )
        
        if response.status_code != 200:
            raise RuntimeError(
                f"llama-server returned {response.status_code}: {response.text}"
            )
        
        result = response.json()
        
        # Extract assistant response
        if "choices" not in result or len(result["choices"]) == 0:
            raise RuntimeError("No response from llama-server")
        
        assistant_content = result["choices"][0]["message"]["content"]
        
        # Add assistant message to history
        assistant_msg = {"role": "assistant", "content": assistant_content}
        self._messages.append(assistant_msg)
        self._ctx_messages.append(assistant_msg)
        
        return assistant_content
    
    def regenerate(
        self,
        new_user_prompt: str,
        **override_params
    ) -> str:
        """Regenerate the response by replacing the last user prompt.
        
        This removes the last user-assistant exchange and replaces it
        with a new prompt. The new exchange is added to msg_history but
        the old one is removed from ctx_history.
        
        Args:
            new_user_prompt: The replacement prompt
            **override_params: Override default generation parameters
            
        Returns:
            The assistant's response to the new prompt
        """
        if len(self._messages) < 2:
            raise ValueError("Cannot regenerate: no previous conversation to replace")
        
        # Remove last user-assistant exchange from both histories
        # (assuming the last message is assistant and second-to-last is user)
        if self._messages[-1]["role"] == "assistant":
            self._messages.pop()
        if self._messages[-1]["role"] == "user":
            self._messages.pop()
        
        # Remove from context history too
        if self._ctx_messages and self._ctx_messages[-1]["role"] == "assistant":
            self._ctx_messages.pop()
        if self._ctx_messages and self._ctx_messages[-1]["role"] == "user":
            self._ctx_messages.pop()
        
        # Send new prompt
        return self.prompt(new_user_prompt, **override_params)
    
    def msg_history(self) -> List[Dict[str, str]]:
        """Get the full message history including regenerated messages.
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self._messages.copy()
    
    def ctx_history(self) -> List[Dict[str, str]]:
        """Get the context history excluding regenerated messages.
        
        This returns only the messages that are part of the current
        conversation context, without messages that were replaced
        by regenerate().
        
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        return self._ctx_messages.copy()
    
    def ctx_stats(self) -> Dict[str, Any]:
        """Get context statistics from the llama-server.
        
        Returns detailed information about context usage, including
        tokens processed, context window size, and available capacity.
        
        Returns:
            Dictionary containing context statistics:
            - ctx_size: Total context window size
            - tokens_used: Number of tokens in current context
            - tokens_available: Remaining context capacity
            - slots: Detailed slot information (if available)
        """
        try:
            # Get slot information
            slots_response = self._session.get(
                f"{self.base_url}/slots",
                timeout=5
            )
            
            if slots_response.status_code == 200:
                slots_data = slots_response.json()
                
                # Parse slot data
                stats = {
                    "slots": slots_data,
                    "ctx_size": None,
                    "tokens_used": None,
                    "tokens_available": None
                }
                
                # Try to extract useful stats from first slot
                if isinstance(slots_data, list) and len(slots_data) > 0:
                    slot = slots_data[0]
                    stats["ctx_size"] = slot.get("n_ctx")
                    
                    # Calculate tokens used (prompt + generated)
                    n_prompt = slot.get("n_prompt_tokens_processed", 0)
                    n_decoded = slot.get("n_decoded", 0)
                    stats["tokens_used"] = n_prompt + n_decoded
                    
                    if stats["ctx_size"]:
                        stats["tokens_available"] = stats["ctx_size"] - stats["tokens_used"]
                
                return stats
            else:
                # Slots endpoint not available, try /props
                props_response = self._session.get(
                    f"{self.base_url}/props",
                    timeout=5
                )
                
                if props_response.status_code == 200:
                    props_data = props_response.json()
                    return {
                        "ctx_size": props_data.get("default_generation_settings", {}).get("n_ctx"),
                        "tokens_used": None,
                        "tokens_available": None,
                        "props": props_data
                    }
                else:
                    return {
                        "error": "Unable to retrieve context stats",
                        "slots_status": slots_response.status_code,
                        "props_status": props_response.status_code
                    }
        
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Failed to retrieve context stats: {e}"
            }
    
    def clear_history(self, keep_system: bool = True):
        """Clear conversation history.
        
        Args:
            keep_system: If True, preserves the system prompt
        """
        if keep_system and self._messages and self._messages[0].get("role") == "system":
            system_msg = self._messages[0]
            self._messages = [system_msg]
            self._ctx_messages = [system_msg]
        else:
            self._messages = []
            self._ctx_messages = []


# Example usage
if __name__ == "__main__":
    # Basic usage
    with LlamaServerSession(
        base_url="http://localhost:8080",
        temperature=0.7,
        max_tokens=100
    ) as session:
        # Set system prompt
        session.system_prompt("You are a helpful assistant.")
        
        # Send a prompt
        response = session.prompt("What is Python?")
        print(f"Response: {response}\n")
        
        # Get context stats
        stats = session.ctx_stats()
        print(f"Context stats: {json.dumps(stats, indent=2)}\n")
        
        # Send another prompt
        response2 = session.prompt("Can you give me an example?")
        print(f"Response 2: {response2}\n")
        
        # View full history
        print("Full message history:")
        for msg in session.msg_history():
            print(f"  {msg['role']}: {msg['content'][:50]}...")
        
        # Regenerate last response with different prompt
        response3 = session.regenerate("Tell me about its history instead")
        print(f"\nRegenerated response: {response3}\n")
        
        # Context history (without the replaced prompt)
        print("Context history (active conversation):")
        for msg in session.ctx_history():
            print(f"  {msg['role']}: {msg['content'][:50]}...")
