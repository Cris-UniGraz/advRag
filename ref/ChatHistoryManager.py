class ChatHistoryManager:
    def __init__(self, max_history_size=10):
        self.history = []
        self.max_history_size = max_history_size
        
    def add_interaction(self, query: str, response: str):
        self.history.append((query, response))
        if len(self.history) > self.max_history_size:
            self.history.pop(0)  # Elimina la interacción más antigua
            
    def get_current_tokens(self) -> int:
        total_text = " ".join([f"{q} {r}" for q, r in self.history])
        return len(total_text.split())  # Estimación simple de tokens
        
    def clear_if_needed(self, max_tokens=6000):
        while self.get_current_tokens() > max_tokens and self.history:
            self.history.pop(0)