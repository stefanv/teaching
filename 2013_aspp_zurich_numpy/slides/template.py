import sys

fn = sys.argv[1]
f = open(sys.argv[1], 'r')
data = f.readlines()

data_out = []
for line in data:
    if line.startswith('{{INCLUDE'):
        inc = line.split(' ')[1].replace('}}', '').strip()
        print "Including ", inc
        with open(inc, 'r') as f:
            data_out.extend(f.readlines())
    else:
        data_out.append(line)


out = open(fn.replace('.tmpl', ''), 'w')
out.writelines(data_out)
