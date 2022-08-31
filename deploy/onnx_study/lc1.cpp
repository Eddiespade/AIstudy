class Trie {
private:
    vector<Trie*> children;
    vector<Trie*> isEnd;

    Trie* find(string &str) {
        Trie* ptr = this;
        for (auto& c:str) {
            int id = c - 'a';
            if (!ptr->children[id]) {
                return nullptr;
            }
            ptr = ptr->children[id];
        }
        return ptr;
    }

public:
    Trie() {
        children(26);
        isEnd(false);
    }
    
    void insert(string word) {
        Trie* ptr = this;
        for (auto& c:word) {
            int id = c - 'a';
            if (!ptr->children[id]) {
                ptr->children[id] = new Trie();
            }
            ptr = ptr->children[id];
        }
        ptr->isEnd = true;
    }
    
    bool search(string word) {
        Trie* node = find(word);
        return node && node->isEnd;
    }
    
    bool startsWith(string prefix) {
        Trie* node = find(word);
        return node;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie* obj = new Trie();
 * obj->insert(word);
 * bool param_2 = obj->search(word);
 * bool param_3 = obj->startsWith(prefix);
 */