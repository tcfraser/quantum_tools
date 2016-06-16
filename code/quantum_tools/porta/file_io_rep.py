import os
from parse import parse
import re
from pprint import pprint
from fractions import Fraction
from . import porta_file_templates
from string import Formatter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

def _get_template_file():
    template_loc = os.path.join(_THIS_DIR, 'template.ieq')
    with open(template_loc, 'r') as template_file:
        template = template_file.read()
    return template

def _remove_text_inside_brackets(text, brackets="()[]"):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0
                break
        else: # character is not a bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)

def _get_example_file_name(name):
    return os.path.join(_THIS_DIR, '{0}.ieq'.format(name))

TERM = re.compile(r'(?!$)(?P<sign>\+|\-|)(?P<coeff>[0-9]+\/[0-9]+|[0-9]+|)(?P<var>[A-Za-z]+[0-9]+|)')
RELATION = re.compile(r'(?P<lhs>.*)(?P<eqstr>\=\=|\>\=|\=\>|\=\<|\<\=)(?P<rhs>.*)')

class Term():

    def __init__(self, sign, fraction, variable):
        self._sign = sign
        self._fraction = fraction
        self._variable = variable

    @classmethod
    def from_tuple(cls, tuple):
        sign, fraction, variable = tuple
        if sign == '':
            sign = '+'
        if not isinstance(fraction, Fraction):
            if fraction == '':
                fraction = Fraction(1)
            else:
                fraction = Fraction(fraction)
        if variable is '':
            variable = None
        return cls(sign, fraction, variable)

    def __str__(self):
        if self._variable is None:
            str_variable = ''
        else:
            str_variable = self._variable
        str_sign = self._sign
        str_fraction = str(self._fraction)
        return str_sign+str_fraction+str_variable

class Terms():

    def __init__(self, terms):
        self._terms = terms

    def __str__(self):
        # for term in self._terms:
        #     print(term)
        return ''.join(str(term) for term in self._terms)

    @classmethod
    def from_string(cls, string):
        parsed_string = _remove_text_inside_brackets(string)
        parsed_string = parsed_string.replace(" ", "")
        term_tuples = TERM.findall(parsed_string)
        # print(term_tuples)
        terms = list(map(Term.from_tuple, term_tuples))
        return Terms(terms)

class Relation():

    def __init__(self, LHS, eqstr, RHS):
        self._LHS = LHS
        self._RHS = RHS
        self._eqstr = eqstr

    def __str__(self):
        return "{LHS} {eq} {RHS}".format(
            LHS=self._LHS,
            RHS=self._RHS,
            eq=self._eqstr,
        )

    @classmethod
    def from_string(cls, string):
        match = RELATION.match(string)
        LHSstr = match.group('lhs')
        RHSstr = match.group('rhs')
        eqstr = match.group('eqstr')
        LHS = Terms.from_string(LHSstr)
        RHS = Terms.from_string(RHSstr)
        return Relation(LHS, eqstr, RHS)

class PortaFile():

    def __init__(self):
        pass

    @classmethod
    def fromObj(cls, obj):
        portaFile = cls()
        for obj_key in obj:
            setattr(portaFile, obj_key, obj[obj_key])
        # print(portaFile.__dict__)
        return portaFile

    @classmethod
    def fromFile(cls, file_name):
        with open(file_name, 'r') as _file:
            read_data = _file.read()
        template = _get_template_file()
        parse_result = parse(template, read_data)
        assert(parse_result is not None), "Failed to parse raw porta file."
        source = parse_result.named
        target = {}
        for key in source:
            target[key] = PortaFile.__str_parse(key, source[key], fromfile = True)
        portaFile = PortaFile.fromObj(target)
        return portaFile

    def __str__(self):
        str_list = []
        fs = _get_template_file()
        fs_kwargs = {}
        # for literal_test, field_name, format_spec, conversion in Formatter.parse(Formatter, fs):
        for _, field_name, _, _ in Formatter.parse(Formatter, fs):
            if field_name is not None and hasattr(self, field_name):
                attr = getattr(self, field_name, None)
                parsed_attr = PortaFile.__str_parse(field_name, attr, tofile = True)
                fs_kwargs[field_name] = parsed_attr
        str_list.append(fs.format(**fs_kwargs))
        return ''.join(str_list)

    def toFile(self, file_name):
        with open(file_name, 'w+') as _file:
            _file.write(str(self))

    @staticmethod
    def __str_parse(key, value, fromfile=False, tofile=False):
        if fromfile == tofile:
            raise Exception("Only one of fromfile and tofile can be true.")
        parsed_value = value # Soon to be changed

        if key == 'dim':
            if fromfile:
                parsed_value = int(value)
            if tofile:
                parsed_value = str(value)
        if key == 'elimination_order':
            if fromfile:
                parsed_value = list(map(int, value.split()))
            if tofile:
                parsed_value = ' '.join(map(str, value))
        if key in ['lower_bounds', 'upper_bounds', 'valid']:
            if fromfile:
                parsed_value = list(map(Fraction, value.split()))
            if tofile:
                parsed_value = ' '.join(map(str, value))
        if key in ['equalities_section', 'inequalities_section']:
            if fromfile:
                parsed_value = parsed_value.split("\n")
                parsed_value = list(map(Relation.from_string, parsed_value))
            if tofile:
                parsed_value = list(map(str, value))
                for i, relation in enumerate(parsed_value):
                    parsed_value[i] = "(  {num}) {rel}".format(num=i+1, rel=relation)
                parsed_value = '\n'.join(parsed_value)
        return parsed_value

def test():
    pf = PortaFile.fromFile(_get_example_file_name('example'))
    # pf.toFile(_get_example_file_name('example_passed_through_python'))
    print(pf)
    # print(_get_template_file())
    # print(Relation.from_string('(  11) 7+3x2+27x1-28x2+57x4-37x5+4/15x2+1/15x5>=2x2+0'))
    # print(_get_template_file())

if __name__ == '__main__':
    test()