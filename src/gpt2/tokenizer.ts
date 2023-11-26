export interface TrieNode {
  children: Record<string, TrieNode>;
  value?: number;
}

export class Tokenizer {
  vocab: string[];
  trie: TrieNode;

  constructor(vocab: Record<string, number>) {
    this.vocab = Array(vocab.length);

    this.trie = {
      children: {},
    };

    const encoder = new TextEncoder();

    Object.entries(vocab).forEach(([token, value]) => {
      let currentNode = this.trie;

      const bytes = encoder.encode(token);

      for (let i = 0; i < bytes.length; i++) {
        const char = String.fromCharCode(bytes[i]);

        if (currentNode.children[char] == null) {
          currentNode.children[char] = {
            children: {},
          };
        }

        currentNode = currentNode.children[char];
      }

      currentNode.value = value;
      this.vocab[value] = token;
    });
  }

  encode(text: string): number[] {
    const tokens: number[] = [];

    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);

    // Depth-first search to find the longest matching token.
    let currentNode = this.trie;
    let tokenStartIndex = 0;

    while (tokenStartIndex < bytes.length) {
      const char = String.fromCharCode(bytes[tokenStartIndex]);

      if (currentNode.children[char] == null) {
        if (currentNode.value != null) {
          tokens.push(currentNode.value);
          currentNode = this.trie;
          continue;
        }

        // If the current character is not in the trie, skip it.
        tokenStartIndex++;
        continue;
      }

      currentNode = currentNode.children[char];
      tokenStartIndex++;
    }

    if (currentNode.value != null) {
      tokens.push(currentNode.value);
    }

    return tokens;
  }

  decode(tokens: number[]): string {
    return tokens.map(token => this.vocab[token]).join('');
  }
}
