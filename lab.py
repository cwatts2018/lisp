"""
6.1010 Spring '23 Lab 11: LISP Interpreter Part 1
"""
#!/usr/bin/env python3

import sys
import doctest

sys.setrecursionlimit(20_000)


# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
  
    new_s = ""
    i = 0
    while i < len(source):
        if source[i] == "(":
            new_s += " ( "
        elif source[i] == ")":
            new_s += " ) "
        elif source[i] == ";":
            try:
                i = i+source[i:].index("\n")+1
            except:
                break
        else:
            new_s += source[i]
        i += 1
    tokens = new_s.split()
    return tokens

  


def parse(tokens):
    
    #check well formed
    open_par = 0
    close_par = 0
    for i in range(len(tokens)):
        elt = tokens[i]
        if elt == "(":
            open_par += 1
        elif elt == ")":
            close_par += 1
        if open_par < close_par:
            raise SchemeSyntaxError
    if open_par != close_par:
        raise SchemeSyntaxError 
    if (tokens[0] != "(" or tokens[len(tokens)-1] != ")") and len(tokens) > 1:
        raise SchemeSyntaxError
    if open_par == 0 and len(tokens) > 1:
        raise SchemeSyntaxError
    
    def actual_parse(tokens):
        """
        Parses a list of tokens, constructing a representation where:
            * symbols are represented as Python strings
            * numbers are represented as Python ints or floats
            * S-expressions are represented as Python lists
    
        Arguments:
            tokens (list): a list of strings representing tokens
        """        
        
                
        def parse_expression(index):
            """
            Parses an individual expression. Returns (parsed_exp, end index of exp)
            """
            def get_parens_end(index):
                """
                Determines the end of an expression by returning the index
                of its closing parentheses.
                """
                open_par = 0
                close_par = 0
                for i in range(index, len(tokens)):
                    elt = tokens[i]
                    if elt == "(":
                        open_par += 1
                    elif elt == ")":
                        close_par += 1
                    if open_par == close_par:
                        return i
                    
            # print('parse expression at index', index)
            end_parens = get_parens_end(index)
            if tokens[index] == "(":
                result = []
                i = index+1
                while i < len(tokens):
                    cur_end = get_parens_end(i)
                    if tokens[i] == "(":
                        result.append(actual_parse(tokens[i:cur_end+1]))
                        i = cur_end + 1
                    elif tokens[i] == ")":
                        i += 1
                    else:
                        result.append(actual_parse(tokens[i:i+1]))
                        i += 1
                return (result, end_parens)
            else:
                # print('third else')
                conv = number_or_symbol(tokens[index])
                # print('return', conv)
                return (conv, index)
        
        # print('parse', tokens)
        if len(tokens) == 0:
            return []
        elif len(tokens) == 1:
            return parse_expression(0)[0]
        else: # loop through and parse exoressions
            result, next_index = parse_expression(0)
        return result
  
    return actual_parse(tokens)


######################
# Built-in Functions #
######################


scheme_builtins = {
    "+": sum,
    "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
    "*": lambda args: mul(args),
    "/": lambda args: div(args)
}


    
def mul(args):
    result = args[0]
    for i in range(1, len(args)):
         result *= args[i] 
    return result

def div(args):
    result = args[0]
    for i in range(1, len(args)):
         result /= args[i] 
    return result


##############
# Evaluation #
##############
class Frame():
    """
    Frame object with a parent frame (None if no parent frame), and bindings.
    """
    def __init__(self, parent_frame=None, bindings = None):
        self.parent = parent_frame
        self.bindings = {}
        if not isinstance(bindings, type(None)):
            self.bindings.update(bindings)
    
    def store_binding(self, name, exp):
        self.bindings[name] = exp
    
    def get_binding(self, key):
        """
        Returns expression binded to key in frame or parent frame, 
        or None if no expression exists.
        """
        # print('here', self.bindings)
        # print(key)
        if key in self.bindings:
            return self.bindings[key]
        if isinstance(self.parent, type(None)):
            # print('herererer')
            return None
        else:
            return self.parent.get_binding(key)   

class Function(Frame):
    """
    Function object is a Frame with a parent frame,
    a list of parameter variable names, and a body expression.
    """
    def __init__(self, params, body, parent_frame=None):
        self.parent = parent_frame #where function defined
        self.bindings = {} #vars and values
        self.params = params #param names
        self.body = body
    
    def get_params(self):
        return self.params
    
    def get_body(self):
        return self.body
    
    def call(self, tree_simplified):
        """
        Func.call(params) calls this function, where tree_simplified is a list of
        the values for the params, and the function evaluates and returns
        the result of the function body expression.
        """
        new_frame = Frame(self.parent)
        if len(tree_simplified) != len(self.params):
            raise SchemeEvaluationError
        for i in range(len(tree_simplified)):
            new_frame.store_binding(self.params[i], tree_simplified[i])
        return evaluate(self.body, new_frame)
    
        
global_frame = Frame(None, scheme_builtins)

def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if isinstance(frame, type(None)):
        frame = Frame(global_frame, None)
        
    if not isinstance(tree, list):
        if isinstance(tree, str): #symbol
            exp = frame.get_binding(tree)  #if symbol is key in frame, return exp
            if isinstance(exp, type(None)):
                raise SchemeNameError
            else:
                return exp
        return tree #num
    else: #list
        #nested list (function w/ initialized params)
        if isinstance(tree[0], list) and (tree[0][0] == "lambda" or not isinstance(frame.get_binding(tree[0][0]), type(None))):
            f = evaluate(tree[0], frame)
            tree_simplified = simplify_tree(tree,frame)
            return f.call(tree_simplified)
        if tree[0] == "lambda":
            vals = tree[3:]
            f = Function(tree[1], tree[2], frame)
            if len(vals) == 0: #if just function, return function, don't store
                return f
        if tree[0] == "define": #storing
            if isinstance(tree[1], list):
                return evaluate(rewrite_function(tree), frame)
            exp = evaluate(tree[2], frame)
            frame.store_binding(tree[1], exp)
            return exp

        #if function name
        func = frame.get_binding(tree[0]) #check if function is stored
        
        #if not stored, raise error
        if isinstance(func, type(None)):
            raise SchemeEvaluationError
        
        #else evaluate function
        tree_simplified = tree_simplified = simplify_tree(tree,frame)
        if isinstance(func, Function):
            return func.call(tree_simplified)
        return func(tree_simplified)
    

def simplify_tree(tree, frame):
    tree_simplified = []
    for i in range(1, len(tree)):
        tree_simplified.append(evaluate(tree[i], frame))
    return tree_simplified

def rewrite_function(tree):
    params = []
    for i in range(1,len(tree[1])):
        params.append(tree[1][i])
    result = ["define", tree[1][0], ["lambda"]]
    result[2].append(params)
    result[2].append(tree[2])
    return result

def result_and_frame(tree, frame=None):
    """
    Returns tuple (result, frame evaluated)
    """
    if isinstance(frame, type(None)):
        frame = Frame(global_frame, None)
    return (evaluate(tree, frame), frame)
    

def repl(verbose=False):
    """
    Read in a single line of user input, evaluate the expression, and print 
    out the result. Repeat until user inputs "QUIT"
    
    Arguments:
        verbose: optional argument, if True will display tokens and parsed
            expression in addition to more detailed error output.
    """
    import traceback
    _, frame = result_and_frame(["+"])  # make a global frame
    while True:
        input_str = input("in> ")
        if input_str == "QUIT":
            return
        try:
            token_list = tokenize(input_str)
            if verbose:
                print("tokens>", token_list)
            expression = parse(token_list)
            if verbose:
                print("expression>", expression)
            output, frame = result_and_frame(expression, frame)
            print("  out>", output)
        except SchemeError as e:
            if verbose:
                traceback.print_tb(e.__traceback__)
            print("Error>", repr(e))

if __name__ == "__main__":
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)
    # n = 18
    # with open(os.path.join(TEST_DIRECTORY, "test_outputs", f"{n:02d}.txt")) as f:
    #     expected = eval(f.read())
    # env = None
    # results = []
    # try:
    #     t = make_tester(result_and_frame)
    # except:
    #     t = make_tester(evaluate)
    # with open(os.path.join(TEST_DIRECTORY, "test_inputs", f"{n:02d}.scm")) as f:
    #     for line in iter(f.readline, ""):
    #         try:
    #             parsed = lab.parse(lab.tokenize(line.strip()))
    #         except lab.SchemeSyntaxError:
    #             print('here')
    #             results.append(
    #                 {
    #                     "expression": line.strip(),
    #                     "ok": False,
    #                     "type": "SchemeSyntaxError",
    #                     "when": "parse",
    #                 }
    #             )
    #             continue
    #         out = t(*((parsed,) if env is None else (parsed, env)))
    #         if out["ok"]:
    #             env = out["output"][1]
    #         if out["ok"]:
    #             try:
    #                 typecheck = (int, float, lab.Pair)
    #                 func = list_from_ll
    #             except:
    #                 typecheck = (int, float)
    #                 func = lambda x: x if isinstance(x, typecheck) else "SOMETHING"
    #             out["output"] = func(out["output"][0])
    #         out["expression"] = line.strip()
    #         results.append(out)
    # for ix, (result, exp) in enumerate(zip(results, expected)):
    #     msg = f"for line {ix+1} in test_inputs/{n:02d}.scm:\n    {result['expression']}"
    #     compare_outputs(result, exp, msg=msg)
        
    x = "(define (spam eggs) (lambda (x y) (- eggs x y)))"
    x = "((spam 9) 8)"
    print(x)
    print(tokenize(x))
    print(parse(tokenize(x)))
    # uncommenting the following line will run doctests from above
    # doctest.testmod()
    # repl(True)
    # l = "(define (square x) (* x x))"
    # a = tokenize(l)
    # b = parse(a)
    # c = rewrite_function(b)
    # print(c)
    # print('heeh' , evaluate(b))
    # z = "(square 21)"
    # y = tokenize(z)
    # x = parse(y)
    # zz = evaluate(x)
    # print('lili', zz)
    # print(a)
    # print(b)
    # print(x)
    # print(rewrite_function(b))
    # s = "(define circle-area\n  (lambda (r) ;hi there \n   (* 3.14 (* r r))\n  )\n)"
    # print(s)
    # s = s.split()
    # print(s)
    # d = "  d  \n     s"
    # d = d.split()
    # x = parse(['(', '+', '2', 'x' ,')'])
    # x = evaluate(['*', 3, 3, 3])
    # output = result_and_frame(5)
    # print(output)
