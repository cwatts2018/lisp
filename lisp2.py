import sys
sys.setrecursionlimit(20_000)

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
                conv = number_or_symbol(tokens[index])
                return (conv, index)
        
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
    "/": lambda args: div(args),
    ">": lambda args: boolean(args, ">"),
    "<": lambda args: boolean(args, "<"),
    ">=": lambda args: boolean(args, ">="),
    "<=": lambda args: boolean(args, "<="),
    "equal?": lambda args: boolean(args, "equal?"),
    "not": lambda args: not_ev(args),
    "#t": True,
    "#f": False,
    "car": lambda arg: get_car(arg),
    "cdr": lambda arg: get_cdr(arg),
    "nil": [],
    "list?": lambda obj: list_q(obj),
    "length": lambda ls: length(ls),
    "list-ref": lambda args: list_ref(args[0], args[1]),
    "append": lambda ls: append(ls),
    "map": lambda args: map_func(args[0], args[1]),
    "filter": lambda args: filt(args[0], args[1]),
    "reduce": lambda args: reduce(args[0], args[1], args[2]),
    "begin": lambda args: begin(args),
    # "del": lambda arg: delete(arg)
}   

def evaluate_file(file_name, frame=None):
    """
    Evaluates a file given file name.
    """
    file = open(file_name)
    tree = parse(tokenize(file.read()))
    file.close()
    return evaluate(tree, frame)
    

def begin(args):
    """
    Evaluates all arguments in args.
    """
    for arg in args:
        result = evaluate(arg)
    return result

def reduce(func, ls, initval):
    """
    Reduces ls by applying func element by element starting with initval.
    """
    if not list_q([ls]):
        raise SchemeEvaluationError("passed in nonlist")
    elts = []
    result = initval
    for i in range(length([ls])):
        elts.append(list_ref(ls, i))
    for elt in elts:
        result = func([result, elt])
    return result
          

def filt(func, ls):
    """
    Filters ls and returns tree of only elements that satisfy func.
    """
    if not list_q([ls]):
        raise SchemeEvaluationError("passed in nonlist")
    result = []
    elts = []
    for i in range(length([ls])):
        elts.append(list_ref(ls, i))
    for elt in elts:
        if func([elt]):
            result.append(elt)
    return create_list(result)
    

def map_func(func, ls):
    """
    Maps the func to ls and returns a new list with func applied to every
    element of ls.
    """
    if not list_q([ls]):
        raise SchemeEvaluationError("passed in nonlist")
    result = []
    elts = []
    for i in range(length([ls])):
        elts.append(list_ref(ls, i))
    for elt in elts:
        result.append(func([elt]))
    if len(result) != 0:
        return create_list(result)
    return result

def append(ls):
    """
    ls is list of lists
    (append (list 1) (list 2 3 4 5 9 10))
    'ok': True, 'output': [1, 2, 3, 4, 5, 9, 10]},
    """
    elts = []
    if len(ls) == 0:
        return []
    for mini_ls in ls:
        if not list_q([mini_ls]):
            raise SchemeEvaluationError("appending a non list")
        if mini_ls == []:
            pass
        else:
            for i in range(length([mini_ls])):
                elts.append(list_ref(mini_ls, i))
    if len(elts) == 0:
        return []
    return evaluate(create_list(elts))

def list_ref(ls, index):
    """
    ls is ls (not in extra list)
    """
    if isinstance(ls, Pair) and not list_q([ls]):
        if index == 0:
            return get_car([ls])
        else:
            raise SchemeEvaluationError("non list")
    cur_ls = ls
    while index > 0:
        cur_ls = get_cdr([cur_ls])
        index -= 1
    return get_car([cur_ls])

def length(ls):
    """
    ls is [ls]. returns length of ls
    """
    if not list_q(ls):
        raise SchemeEvaluationError("input not a list")
    length = 0
    cur_ls = ls[0]
    while cur_ls != []:
        cur_ls = get_cdr([cur_ls])
        length += 1
    return length
    
def list_q(obj):
    """
    obj is [object]. returns whether obj is a list
    """
    if len(obj) == 1 and obj[0] == []:
        return True
    elif isinstance(obj[0], Pair) and list_q([get_cdr([obj[0]])]):
        return True
    return False

def get_car(arg):
    """
    arg is [object]. returns car of arg
    """
    if (len(arg) != 1) or not isinstance(arg[0], Pair):
        raise SchemeEvaluationError
    return arg[0].car

def get_cdr(arg):
    """
    arg is [object]. returns cdr of arg
    """
    if (len(arg) != 1) or not isinstance(arg[0], Pair):
        raise SchemeEvaluationError
    return arg[0].cdr
    
def mul(args):
    """
    Multiplies elements in args by each other.
    """
    result = args[0]
    for i in range(1, len(args)):
        result *= args[i]
    return result

def div(args):
    """
    Divides elements in args by each other.
    """
    result = args[0]
    for i in range(1, len(args)):
        result /= args[i]
    return result

def boolean(args, sign):
    """
    Evaluates boolean statements given arguments and sign.
    """
    for i in range(len(args)-1):
        if args[i] != args[i+1] and sign == "equal?":
            return False
        elif args[i] <= args[i+1] and sign == ">":
            return False
        elif args[i] >= args[i+1] and sign == "<":
            return False
        elif args[i] > args[i+1] and sign == "<=":
            return False
        elif args[i] < args[i+1] and sign == ">=":
            return False       
    return True

def not_ev(args):
    """
    Evaluates the opposite of the truth of args.
    """
    if len(args) != 1:
        raise SchemeEvaluationError
    if args[0]:
        return False
    return True


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
        self.old_bindings = []
        if not isinstance(bindings, type(None)):
            self.bindings.update(bindings)
        if not isinstance(parent_frame, type(None)):
            self.old_bindings = parent_frame.old_bindings[:]
    
    def store_binding(self, name, exp):
        """
        Stores exp as name in frame's bindings.
        """
        self.bindings[name] = exp
    
    def get_binding(self, key):
        """
        Returns expression binded to key in frame or parent frame, 
        or None if no expression exists.
        """
        if key in self.bindings:
            return self.bindings[key]
        elif key in self.old_bindings:
            if not isinstance(self.parent, type(None)):
                try:
                    val = self.parent.get_binding(key) 
                    return val
                except:
                    return "special case error"
            else:
                raise "special case error"
        elif isinstance(self.parent, type(None)):
            raise SchemeNameError
        else:
            return self.parent.get_binding(key) 
        
    def remove_binding(self, key):
        """
        Removes binding at key and returns old value.
        """
        if key in self.bindings:
            self.old_bindings.append(key)
            return self.bindings.pop(key)
        else: 
            raise SchemeNameError
            
    def set_binding(self, key, val):
        """
        Returns expression binded to key in frame or parent frame, 
        or None if no expression exists.
        """
        if key in self.bindings:
            self.bindings[key] = val
            return val
        elif isinstance(self.parent, type(None)):
            raise SchemeNameError
        else:
            return self.parent.set_binding(key, val) 


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
        # print('created function', self, 'with bindings', self.bindings)
    
    def get_params(self):
        return self.params
    
    def get_body(self):
        return self.body
    
    def __call__(self, tree_simplified):
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
    
    def __str__(self):
        return "Function(" + str(self.params) + ", " + str(self.body) + ")"

class Pair():
    """
    Represents a Pair object with a car and cdr.
    """
    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr
        
    def __str__(self):
        return "Pair(" + str(self.car) + ", " + str(self.cdr) + ")"    

global_frame = Frame(None, scheme_builtins)

def create_list(params, frame=None):
    """
    Creates a Pair object from a list of params.
    """
    if len(params) == 0:
        return []
    length = len(params)
    eval_params = []
    for par in params:
        if par == []:
            eval_params.append([])
        else:
            eval_params.append(evaluate(par, frame))
        
    result = Pair(eval_params[length-1], [])
    for i in range(length-2, -1, -1):
        if eval_params[i] == []:
            result = Pair([], result)
        else:
            result = Pair(eval_params[i], result)
    return result  

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
            if exp == "special case error":
                raise SchemeNameError
            else:
                return exp
        return tree #num
    else: #list
        if len(tree) == 0: #empty list
            raise SchemeEvaluationError
        #nested list (function w/ initialized params)
        if isinstance(tree[0], list):
            tree_simplified = simplify_tree(tree,0,frame)
            return evaluate(tree_simplified, frame)
        elif tree[0] == "lambda":
            vals = tree[3:]
            f = Function(tree[1], tree[2], frame)
            if len(vals) == 0: #if just function, return function, don't store
                return f
        elif tree[0] == "define": #storing
            if isinstance(tree[1], list):
                return evaluate(rewrite_function(tree), frame)
            exp = evaluate(tree[2], frame)
            frame.store_binding(tree[1], exp)
            return exp
        elif tree[0] == "if": #(if PRED TRUE_EXP FALSE_EXP)
            if evaluate(tree[1], frame):
                return evaluate(tree[2], frame)
            else:
                return evaluate(tree[3], frame)
        elif tree[0] == "and":
            for i in range(1, len(tree)):
                if not evaluate(tree[i], frame): #if false
                    return False
            return True
        elif tree[0] == "or":
            for i in range(1, len(tree)):
                if evaluate(tree[i], frame): #if true
                    return True
            return False
        elif tree[0] == "cons":
            if len(tree[1:]) != 2:
                raise SchemeEvaluationError
            return Pair(evaluate(tree[1], frame), evaluate(tree[2], frame))
        elif tree[0] == "list":
            if len(tree[1:]) == 0:
                return []
            return create_list(tree[1:], frame)
        elif tree[0] == "set!":
            exp = evaluate(tree[2], frame)
            return frame.set_binding(tree[1], exp)
        elif tree[0] == "let":
            new_frame = Frame(frame)
            for pair in tree[1]:
                val = evaluate(pair[1], frame)
                new_frame.store_binding(pair[0], val)
            return evaluate(tree[2], new_frame)
        elif tree[0] == "del":
            return frame.remove_binding(tree[1])
        elif isinstance(tree[0], Function):
            return tree[0](tree[1:])
        else:
            try:
                return tree[0](tree[1:])
            except:
                pass

        #if function name
        if not isinstance(tree[0], str):
            raise SchemeEvaluationError
        try:
            func = frame.get_binding(tree[0]) #check if function is stored
            if func == "special case error":
                raise SchemeNameError
            tree_simplified = simplify_tree(tree, 1, frame)
            return func(tree_simplified)
        except SchemeNameError:
            raise SchemeNameError
        except:
            raise SchemeEvaluationError

def simplify_tree(tree, start, frame):
    """
    Given a tree list, a starting index, and a frame, evaluates each index
    from start and returns simplified tree.
    """
    tree_simplified = []
    for i in range(start, len(tree)):
        tree_simplified.append(evaluate(tree[i], frame))
    return tree_simplified

def rewrite_function(tree):
    """
    Rewrites a simplified function in Scheme to be inputtable into evaluate.
    """
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

def repl(verbose=False, file_name=None):
    """
    Read in a single line of user input, evaluate the expression, and print 
    out the result. Repeat until user inputs "QUIT"
    
    Arguments:
        verbose: optional argument, if True will display tokens and parsed
            expression in addition to more detailed error output.
    """
    import traceback
    _, frame = result_and_frame(["+"])  # make a global frame
    if not isinstance(file_name, type(None)):
        evaluate_file(file_name, frame)
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
    doctest.testmod()

    #example usage:
    x = "(begin (define (contains? list_ elt) (if (equal? nil list_) #f (if (equal? (car list_) elt) #t (contains? (cdr list_) elt)))) (if (contains? (list 1 2 3) 2) 1 0))"
    y = evaluate(parse(tokenize(x)))
    print(y)
    # repl(True)


