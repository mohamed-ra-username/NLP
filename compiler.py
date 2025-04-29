import re

# Token specification
spec = [
    ('NUMBER', r'\d+(\.\d+)?'),
    ('ID', r'[A-Za-z_]\w*'),
    ('OP', r'==|!=|<=|>=|<|>|[+\-*/=!]'),
    ('KEYWORD', r'\b(if|then|else)\b'),
    ('SKIP', r'[ \t]+'),
    ('LPAREN', r'\$\$'),
    ('RPAREN', r'\$\$'),
    ('NEWLINE', r'\n'),
    ('MISMATCH', r'.'),
]

tok_re = re.compile('|'.join(f'(?P<{n}>{p})' for n, p in spec))

def tokenize(code):
    tokens = []
    for m in tok_re.finditer(code):
        kind = m.lastgroup
        val = m.group()
        if kind == 'NUMBER':
            val = float(val) if '.' in val else int(val)
            tokens.append(('NUM', val))
        elif kind == 'ID':
            if val in ('if', 'then', 'else'):
                tokens.append(('KEYWORD', val))
            else:
                tokens.append(('ID', val))
        elif kind == 'OP':
            tokens.append(('OP', val))
        elif kind == 'KEYWORD':
            tokens.append(('KEYWORD', val))
        elif kind in ('LPAREN', 'RPAREN'):
            tokens.append((kind, val))
        elif kind in ('NEWLINE', 'SKIP'):
            continue
        else:
            raise RuntimeError(f'Unexpected {val}')
    return tokens

class Parser:
    def __init__(self, tks): self.tks, self.i = tks, 0
    def peek(self): return self.tks[self.i] if self.i < len(self.tks) else None
    def consume(self, *types):
        t = self.peek()
        if t and t[0] in types:
            self.i += 1
            return t
        raise RuntimeError(f'Expected {types}, got {t}')

    def parse_expr(self):
        return self.parse_equality()

    def parse_equality(self):
        node = self.parse_term()
        while True:
            t = self.peek()
            if t and t[0] == 'OP' and t[1] in ('==','!=','<','>','<=','>='):
                op = t[1]
                self.consume('OP')
                node = ('binop', op, node, self.parse_term())
            else:
                break
        return node

    def parse_term(self):
        node = self.parse_factor()
        while True:
            t = self.peek()
            if t and t[0] == 'OP' and t[1] in ('+','-'):
                op = t[1]
                self.consume('OP')
                node = ('binop', op, node, self.parse_factor())
            else:
                break
        return node

    def parse_factor(self):
        node = self.parse_primary()
        while True:
            t = self.peek()
            if t and t[0] == 'OP' and t[1] in ('*','/'):
                op = t[1]
                self.consume('OP')
                node = ('binop', op, node, self.parse_primary())
            else:
                break
        return node

    def parse_primary(self):
        t = self.peek()
        if not t:
            raise RuntimeError('Unexpected EOF')
        if t[0]=='NUM':
            self.consume('NUM')
            return ('num', t[1])
        elif t[0]=='ID':
            self.consume('ID')
            return ('var', t[1])
        elif t[0]=='LPAREN':
            self.consume('LPAREN')
            e = self.parse_expr()
            self.consume('RPAREN')
            return e
        elif t[0]=='KEYWORD' and t[1]=='if':
            return self.parse_if()
        else:
            raise RuntimeError(f'Unexpected token {t}')

    def parse_statement(self):
        t = self.peek()
        if t and t[0]=='ID':
            var = t[1]
            self.consume('ID')
            self.consume('OP')  # '='
            expr = self.parse_expr()
            return ('assign', var, expr)
        elif t and t[0]=='KEYWORD' and t[1]=='if':
            return self.parse_if()
        else:
            raise RuntimeError(f'Unknown statement {t}')

    def parse_if(self):
        self.consume('KEYWORD')  # 'if'
        cond = self.parse_expr()
        self.consume('KEYWORD')  # 'then'
        then_stmt = self.parse_statement()
        else_stmt = None
        if self.peek() and self.peek()[0]=='KEYWORD' and self.peek()[1]=='else':
            self.consume('KEYWORD')
            else_stmt = self.parse_statement()
        return ('if', cond, then_stmt, else_stmt)

    def parse_all(self):
        stmts = []
        while self.i < len(self.tks):
            stmts.append(self.parse_statement())
        return stmts

class CodeGen:
    def __init__(self):
        self.vars = {}
        self.code = []
        self.reg_count = 0
        self.label_count = 0

    def new_reg(self): self.reg_count +=1; return f'R{self.reg_count}'
    def new_label(self): self.label_count+=1; return f'L{self.label_count}'

    def gen_expr(self, node):
        t = node[0]
        if t=='num':
            r=self.new_reg()
            self.code.append(f'LOAD_CONST {node[1]} -> {r}')
            return r
        elif t=='var':
            r=self.new_reg()
            self.code.append(f'LOAD {node[1]} -> {r}')
            return r
        elif t=='binop':
            op = node[1]
            l = self.gen_expr(node[2])
            r = self.gen_expr(node[3])
            dest = self.new_reg()
            if op=='+': self.code.append(f'ADD {l} {r} -> {dest}')
            elif op=='-': self.code.append(f'SUB {l} {r} -> {dest}')
            elif op=='*': self.code.append(f'MUL {l} {r} -> {dest}')
            elif op=='/': self.code.append(f'DIV {l} {r} -> {dest}')
            elif op in ('==','!=','<','>','<=','>='):
                self.code.append(f'CMP {l} {r}')
                self.code.append(f'SET {dest} based on {op}')
            return dest

    def gen_stmt(self, node):
        ntype = node[0]
        if ntype=='assign':
            var, expr = node[1], node[2]
            reg = self.gen_expr(expr)
            self.code.append(f'STORE {reg} -> {var}')
            self.vars[var]=reg
        elif ntype=='if':
            cond, then_stmt, else_stmt = node[1], node[2], node[3]
            c_reg = self.gen_expr(cond)
            lbl_else = self.new_label()
            lbl_end = self.new_label()
            self.code.append(f'JZ {c_reg} {lbl_else}')
            self.gen_stmt(then_stmt)
            self.code.append(f'JMP {lbl_end}')
            self.code.append(f'LABEL {lbl_else}')
            if else_stmt:
                self.gen_stmt(else_stmt)
            self.code.append(f'LABEL {lbl_end}')

    def generate(self, stmts):
        for s in stmts:
            self.gen_stmt(s)

def main():
    print("Enter code lines. Type 'END' to finish.")
    lines = []
    while True:
        l = input()
        if l.strip().upper()=='END': break
        lines.append(l)
    code = '\n'.join(lines)
    
    # Step 1: Tokenization
    tokens = tokenize(code)
    print("\nTokens:")
    for t in tokens:
        print(t)
    
    # Step 2: Parsing
    parser = Parser(tokens)
    try:
        stmts = parser.parse_all()
        print("\nParsed AST:")
        from pprint import pprint
        pprint(stmts)
    except RuntimeError as e:
        print('Parse error:', e)
        return
    
    # Step 3: Code Generation
    cg = CodeGen()
    cg.generate(stmts)
    print("\nGenerated Assembly:")
    for line in cg.code:
        print(line)
    print("\nVariable mappings:")
    for var, reg in cg.vars.items():
        print(f"{var} -> {reg}")

if __name__=='__main__': 
    main()
