from dataclasses import dataclass
from utils import lit2i, big2i
from const import EtherType, IpType
from datatypes import bit1, bit2, bit4, bit6, bit13, bit15, bit20, byte1, \
        byte2, byte4, byte6, byte8, byte16
from typing import Union, Optional


@dataclass
class TCPOptions:
    has_max_seg: bit1
    max_seg: byte2
    has_max_win: bit1
    win_scale: byte1
    has_sack_perm: bit1
    has_sack: bit1
    nsack: byte1
    sack: list[tuple[byte4, byte4]]
    has_ts: bit1
    ts: byte4
    prev_ts: byte4
    has_user_to: bit1
    user_timeout_gran: bit1
    user_timeout: bit15


@dataclass
class TCP:
    src: byte2
    dst: byte2
    seq_no: byte4
    ack_no: byte4
    data_off: bit4
    reserved: bit4
    cwr: bit1
    ece: bit1
    urg: bit1
    ack: bit1
    psh: bit1
    rst: bit1
    syn: bit1
    fin: bit1
    window: byte2
    checksum: byte2
    urgent_pointer: byte2
    options: TCPOptions


@dataclass
class UDP:
    src: byte2
    dst: byte2
    len: byte2
    checksum: byte2


@dataclass
class IPv4:
    version: bit4
    ihl: bit4
    dscp: bit6
    ecn: bit2
    total_length: byte2
    id_: byte2
    reserved: bit1
    df: bit1
    mf: bit1
    fragment_offset: bit13
    ttl: byte1
    proto: byte1
    header_checksum: byte2
    src: byte4
    dst: byte4
    options: bytearray
    payload: Union[TCP, UDP]


@dataclass
class IPv6:
    version: bit4
    ds: bit6
    ecn: bit2
    flow: bit20
    length: byte2
    proto: byte1
    hop_limit: byte1
    src: byte16
    dst: byte16
    payload: Union[TCP, UDP]


@dataclass
class EthernetII:
    dst_mac: byte6
    src_mac: byte6
    has_header_802_1q: bit1
    header_802_1q: byte4
    ethertype: byte2
    payload: Union[IPv4, IPv6]


@dataclass
class PCAP:
    timestamp: byte8
    len_captured: byte4
    len_original: byte4
    payload: EthernetII


class Decoder:
    def process(self, p):
        p = bytearray(p)

        ts = lit2i(p[0:4]) * 2**32 + lit2i(p[4:8])
        cpl = lit2i(p[8:12])
        opl = lit2i(p[12:16])
        pcap = PCAP(ts, cpl, opl, None)

        eth_off = 16

        # Eth
        eth_dst = big2i(p[eth_off+0:eth_off+6])
        eth_src = big2i(p[eth_off+6:eth_off+12])
        il_off = eth_off + 14
        ether_type = big2i(p[il_off-2:il_off])
        if ether_type == EtherType.VLAN:

            il_off += 4
            has_header_802_1q = 1
            header_802_1q = big2i(p[il_off-6:il_off-2])
            ether_type = big2i(p[il_off-2:il_off])
            pcap.payload = EthernetII(eth_dst, eth_src, has_header_802_1q,
                                      header_802_1q, ether_type, None)
        else:
            has_header_802_1q = 0
            header_802_1q = 0
            pcap.payload = EthernetII(eth_dst, eth_src, has_header_802_1q,
                                      header_802_1q, ether_type, None)

        # Internet layer
        if ether_type == EtherType.IPv4:
            ip_vers = p[il_off] // 16
            ip_ihl = p[il_off] % 16
            tl_off = il_off + 4 * ip_ihl
            ip_dscp = p[il_off + 1] // 4
            ip_ecn = p[il_off + 1] % 4
            ip_proto = p[il_off+9]

            tl_len = big2i(p[il_off+2:il_off+4]) - 4*ip_ihl
            ip_id = big2i(p[il_off+4:il_off+6])
            ip_reserved = p[il_off+6] // 128
            ip_df = p[il_off+6] // 64 % 2
            ip_mf = p[il_off+6] // 32 % 2
            ip_fo = big2i(p[il_off+6:il_off+7]) % 8192
            ip_ttl = p[il_off+8]
            ip_cs = big2i(p[il_off+10:il_off+12])

            ip_addr_off = il_off + 12
            ip_src = big2i(p[ip_addr_off+0:ip_addr_off+4])
            ip_dst = big2i(p[ip_addr_off+4:ip_addr_off+8])

            pcap.payload.payload = IPv4(ip_vers, ip_ihl, ip_dscp, ip_ecn,
                                        tl_len, ip_id, ip_reserved, ip_df,
                                        ip_mf, ip_fo, ip_ttl, ip_proto,
                                        ip_cs, ip_src, ip_dst, None, None)

        elif ether_type == EtherType.IPv6:
            tl_len = big2i(p[il_off+4:il_off+6])
            ip_vers = p[il_off] // 0x10
            ip_ds = (big2i(p[il_off:il_off+2]) // 0x40) % 0x40
            ip_ecn = (big2i(p[il_off:il_off+2]) // 0x08) % 0x04
            ip_flow = big2i(p[il_off+1:il_off+4]) % 0x200000
            ip_hop_limit = p[il_off+7]
            ip_proto = p[il_off+6]
            ip_addr_off = il_off + 8
            ip_src = big2i(p[ip_addr_off+0:ip_addr_off+16])
            ip_dst = big2i(p[ip_addr_off+16:ip_addr_off+32])
            tl_off = il_off + 40
            pcap.payload.payload = IPv6(ip_vers, ip_ds, ip_ecn, ip_flow,
                                        tl_len, ip_proto, ip_hop_limit, ip_src,
                                        ip_dst, None)

        else:
            ...
            return
            #raise NotImplementedError(
            #    f'No support for EtherType {ether_type:04x}')

        if ip_proto == IpType.TCP:

            tcp_data_off = p[tl_off + 12] // 16
            payload_off = tl_off + 4 * (tcp_data_off)
            tcp_src = big2i(p[tl_off:tl_off+2])
            tcp_dst = big2i(p[tl_off+2:tl_off+4])

            tcp_data_off = p[tl_off + 12] // 16
            tcp_reserved = p[tl_off + 12] % 16
            tcp_seq = big2i(p[tl_off+4:tl_off+8])
            tcp_ack = big2i(p[tl_off+8:tl_off+12])
            tcp_win = big2i(p[tl_off+14:tl_off+16])
            tcp_flags = p[tl_off+13]
            tcp_flag_cwr = (tcp_flags & 0x80) // 0x80
            tcp_flag_ece = (tcp_flags & 0x40) // 0x40
            tcp_flag_urg = (tcp_flags & 0x20) // 0x20
            tcp_flag_ack = (tcp_flags & 0x10) // 0x10
            tcp_flag_psh = (tcp_flags & 0x08) // 0x08
            tcp_flag_rst = (tcp_flags & 0x04) // 0x04
            tcp_flag_syn = (tcp_flags & 0x02) // 0x02
            tcp_flag_fin = tcp_flags & 0x01
            tcp_checksum = big2i(p[tl_off+16:tl_off+18])
            tcp_urg_ptr = big2i(p[tl_off+18:tl_off+20])
            options = TCPOptions(0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0)
            options.sack = []
            i = tl_off + 20
            while i < payload_off:
                opt_kind = p[i]
                if opt_kind == 0:
                    i += 1
                elif opt_kind == 1:
                    i += 1
                elif opt_kind == 2:
                    tcp_max_seg_size = big2i(p[i+2:i+4])
                    options.max_seg = tcp_max_seg_size
                    options.has_max_seg = 1
                    i += 4
                elif opt_kind == 3:
                    tcp_max_window_scale = p[i+2]
                    options.win_scale = tcp_max_window_scale
                    options.has_max_win = 1
                    i += 3
                elif opt_kind == 4:
                    options.sack_perm = 1
                    i += 2
                elif opt_kind == 5:
                    tcp_nsack = p[i+1]
                    tcp_sack = list()
                    for n in range((tcp_nsack - 2) // 8):
                        tcp_sack.append((big2i(p[i+2+n*8:i+6+n*8]),
                                         big2i(p[i+6+n*8:i+10+n*8])))
                    options.nsack = tcp_nsack
                    options.sack = tcp_sack
                    options.has_sack = 1
                    i += p[i+1]
                elif opt_kind == 8:
                    tcp_ts = big2i(p[i+2:i+6])
                    tcp_prev_ts = big2i(p[i+6:i+10])
                    options.ts = tcp_ts
                    options.prev_ts = tcp_prev_ts
                    options.has_ts = 1
                    i += 10
                elif opt_kind == 28:
                    tcp_has_gran = p[i+2] // 0x80
                    tcp_gran = big2i(p[i+2:i+3]) % 0x80
                    options.user_timeout_gran = tcp_has_gran
                    options.user_timeout = tcp_gran
                    options.has_user_to = 1
                    i += 4
                else:
                    assert False, f'unknown case {opt_kind}'
            pcap.payload.payload.payload = TCP(tcp_src, tcp_dst, tcp_seq,
                                               tcp_ack, tcp_data_off,
                                               tcp_reserved, tcp_flag_cwr,
                                               tcp_flag_ece, tcp_flag_urg,
                                               tcp_flag_ack, tcp_flag_psh,
                                               tcp_flag_rst, tcp_flag_syn,
                                               tcp_flag_fin, tcp_win,
                                               tcp_checksum, tcp_urg_ptr,
                                               options)

        elif ip_proto == IpType.UDP:
            udp_src = big2i(p[tl_off:tl_off+2])
            udp_dst = big2i(p[tl_off+2:tl_off+4])
            udp_len = big2i(p[tl_off+4:tl_off+6])
            udp_chksum = big2i(p[tl_off+6:tl_off+8])
            pcap.payload.payload.payload = UDP(udp_src, udp_dst, udp_len,
                                               udp_chksum)

        else:
            ...

        return pcap
