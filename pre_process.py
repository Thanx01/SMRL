import os
import pickle
from collections import defaultdict


ENTITY_PREFIX = '<http://www.worldkg.org/resource/'
RDF_TYPE_REL = '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>'
RDFS_LABEL_REL = '<http://www.w3.org/2000/01/rdf-schema#label>'
WORLDKG_SCHEMA_PREFIX = '<http://www.worldkg.org/schema/'


def split_triple(line):
    line = line.strip()
    if not line:
        return None

    delimiter = '*' if line.count('*') >= 2 else '^'
    parts = [part.strip() for part in line.split(delimiter, 2)]
    if len(parts) != 3 or any(part == '' for part in parts):
        raise ValueError('Invalid triple line: {}'.format(line))
    return parts


def read_triples(path):
    triples = []
    with open(path, encoding='utf-8') as file:
        for line in file:
            triple = split_triple(line)
            if triple is not None:
                triples.append(triple)
    return triples


def is_entity(value):
    return value.startswith(ENTITY_PREFIX)


def is_attribute_relation(relation):
    return (relation == RDF_TYPE_REL
            or relation == RDFS_LABEL_REL
            or relation == '<RegionId>'
            or relation == 'RegionId'
            or relation.startswith(WORLDKG_SCHEMA_PREFIX))


def is_attribute_triple(triple):
    head, relation, tail = triple
    return (not is_entity(head)
            or not is_entity(tail)
            or is_attribute_relation(relation))


def split_relation_attribute_triples(triples):
    relation_triples = []
    attribute_triples = []
    for triple in triples:
        if is_attribute_triple(triple):
            attribute_triples.append(triple)
        else:
            relation_triples.append(triple)
    return relation_triples, attribute_triples


def reidx(tri):
    tri_reidx = []
    ent_reidx = dict()
    rel_reidx = dict()
    for h, r, t in tri:
        if h not in ent_reidx:
            ent_reidx[h] = len(ent_reidx)
        if t not in ent_reidx:
            ent_reidx[t] = len(ent_reidx)
        if r not in rel_reidx:
            rel_reidx[r] = len(rel_reidx)
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(rel_reidx), dict(ent_reidx)


def reidx_withr(tri, rel_reidx):
    tri_reidx = []
    ent_reidx = dict()
    for h, r, t in tri:
        if r not in rel_reidx:
            continue
        if h not in ent_reidx:
            ent_reidx[h] = len(ent_reidx)
        if t not in ent_reidx:
            ent_reidx[t] = len(ent_reidx)
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx, dict(ent_reidx)


def reidx_withr_ande(tri, rel_reidx, ent_reidx):
    tri_reidx = []
    for h, r, t in tri:
        if h not in ent_reidx or t not in ent_reidx or r not in rel_reidx:
            continue
        tri_reidx.append([ent_reidx[h], rel_reidx[r], ent_reidx[t]])
    return tri_reidx


def build_entity_attrs(attribute_triples, ent_reidx, attr_reidx):
    entity_attrs = defaultdict(list)
    for head, relation, tail in attribute_triples:
        if head not in ent_reidx:
            continue
        attr_key = (relation, tail)
        if attr_key not in attr_reidx:
            attr_reidx[attr_key] = len(attr_reidx)
        entity_attrs[ent_reidx[head]].append(attr_reidx[attr_key])

    return {ent_id: sorted(set(attr_ids)) for ent_id, attr_ids in entity_attrs.items()}


def get_data_root(data_root=None):
    if data_root is not None:
        return data_root
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def resolve_ind_data_name(data_name, data_root, ind_data_name=None):
    if ind_data_name is not None:
        return ind_data_name

    candidates = [
        '{}_ind'.format(data_name.replace('_v', '_')),
        '{}_ind'.format(data_name.replace('region_v', 'region_')),
        '{}_ind'.format(data_name),
    ]
    for candidate in candidates:
        if os.path.isdir(os.path.join(data_root, candidate)):
            return candidate
    return candidates[0]


def load_split(data_root, data_name, split):
    return read_triples(os.path.join(data_root, data_name, '{}.txt'.format(split)))


def load_graph_splits(data_root, data_name):
    raw = {split: load_split(data_root, data_name, split) for split in ['train', 'valid', 'test']}
    relation = dict()
    attribute = dict()
    for split, triples in raw.items():
        relation[split], attribute[split] = split_relation_attribute_triples(triples)
    return relation, attribute


def data2pkl(data_name, ind_data_name=None, data_root=None):
    data_root = get_data_root(data_root)
    ind_data_name = resolve_ind_data_name(data_name, data_root, ind_data_name)

    train_rel, train_attr = load_graph_splits(data_root, data_name)
    ind_rel, ind_attr = load_graph_splits(data_root, ind_data_name)

    train_tri, fix_rel_reidx, train_ent_reidx = reidx(train_rel['train'])
    valid_tri = reidx_withr_ande(train_rel['valid'], fix_rel_reidx, train_ent_reidx)
    test_tri = reidx_withr_ande(train_rel['test'], fix_rel_reidx, train_ent_reidx)

    ind_train_tri, ind_ent_reidx = reidx_withr(ind_rel['train'], fix_rel_reidx)
    ind_valid_tri = reidx_withr_ande(ind_rel['valid'], fix_rel_reidx, ind_ent_reidx)
    ind_test_tri = reidx_withr_ande(ind_rel['test'], fix_rel_reidx, ind_ent_reidx)

    attr_reidx = dict()
    train_entity_attrs = build_entity_attrs(train_attr['train'], train_ent_reidx, attr_reidx)
    ind_entity_attrs = build_entity_attrs(ind_attr['train'], ind_ent_reidx, attr_reidx)

    save_data = {
        'data_name': data_name,
        'ind_data_name': ind_data_name,
        'train_graph': {
            'train': train_tri,
            'valid': valid_tri,
            'test': test_tri,
            'entity_attrs': train_entity_attrs,
        },
        'ind_test_graph': {
            'train': ind_train_tri,
            'valid': ind_valid_tri,
            'test': ind_test_tri,
            'entity_attrs': ind_entity_attrs,
        },
        'rel_reidx': fix_rel_reidx,
        'train_ent_reidx': train_ent_reidx,
        'ind_ent_reidx': ind_ent_reidx,
        'attr_reidx': dict(attr_reidx),
        'num_attr': len(attr_reidx),
    }

    with open(os.path.join(data_root, '{}.pkl'.format(data_name)), 'wb') as file:
        pickle.dump(save_data, file)

    return save_data